#include "rnx_extract.h"
#include "utils.h"
#include "log.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <iostream>

extern "C" {
#include "rtklib.h"
}

// 兼容不同 RTKLIB 的 SNR 字段名
#ifndef SNR_FIELD
#define SNR_FIELD SNR
#endif

static void time_to_str(gtime_t t, char *buf, size_t n) {
    double ep[6]; time2epoch(t, ep);
    std::snprintf(buf, n, "%04d-%02d-%02d %02d:%02d:%06.3f",
                  (int)ep[0], (int)ep[1], (int)ep[2],
                  (int)ep[3], (int)ep[4], ep[5]);
}

static int read_one_rinex(const char *path, obs_t *obs, nav_t *nav, sta_t *sta) {
    gtime_t ts=(gtime_t){0}, te=(gtime_t){0};
    double  tint=0.0;
    int     rcv=0;
    const char *opt="";
    int ret = readrnxt(path, rcv, ts, te, tint, opt, obs, nav, sta);
    return ret;
}

static void print_csv_header(FILE *fo,
                             const std::vector<unsigned char>& codes,
                             const std::unordered_map<unsigned char,std::string>& code2label) {
    // 固定列
    std::fprintf(fo, "time_utc,sat,station,rec_x,rec_y,rec_z");
    // 动态追加各信号列（列名即 "Sxx"）
    for (auto c : codes) {
        auto it = code2label.find(c);
        const std::string name = (it==code2label.end() ? "S??" : it->second);
        std::fprintf(fo, ",%s", name.c_str());
    }
    std::fprintf(fo, "\n");
}

void extract_rnx_to_csvs(const std::filesystem::path& rnx_path,
                         const std::filesystem::path& out_root,
                         const std::string& station_tag,
                         const std::vector<unsigned char>& expected_codes_vec,
                         const std::unordered_map<unsigned char, std::string>& code2label) {
    if (expected_codes_vec.empty()) {
        LOG_WARN("expected_codes is empty; nothing to do. file=%s", rnx_path.string().c_str());
        return;
    }
    std::unordered_set<unsigned char> expected_codes(expected_codes_vec.begin(), expected_codes_vec.end());

    obs_t obs={0}; nav_t nav={0}; sta_t sta={0};
    if (!read_one_rinex(rnx_path.string().c_str(), &obs, &nav, &sta)) {
        LOG_ERROR("READ RNX FAILED: %s", rnx_path.string().c_str());
        return;
    }

    // 记录每颗卫星看到了哪些信号，用于缺失告警
    std::map<std::string, std::set<unsigned char>> seen_by_sat;

    // 文件为：<out>/<year>/<station>.csv  （每个 RINEX → 1 文件）
    FILE* fo = nullptr;
    int current_year = -1;
    auto ensure_sink = [&](int year)->FILE*{
        if (fo && year == current_year) return fo;

        // 年度变化则换文件夹（一般同一 RINEX 不跨年，但仍做健壮处理）
        if (fo) { std::fclose(fo); fo = nullptr; }
        auto dir = out_root / std::to_string(year);
        ensure_dir(dir);

        auto fname = dir / (station_tag + std::string(".csv"));
        const bool new_file = !std::filesystem::exists(fname);
        fo = std::fopen(fname.string().c_str(), new_file ? "wb" : "ab");
        if (!fo) {
            LOG_ERROR("OPEN CSV FAILED: %s", fname.string().c_str());
            return (FILE*)nullptr;
        }
        static const size_t BUFSZ = 1<<20;
        setvbuf(fo, nullptr, _IOFBF, BUFSZ);

        if (new_file) {
            print_csv_header(fo, expected_codes_vec, code2label);
        }
        current_year = year;
        return fo;
    };

    for (int i=0;i<obs.n;++i){
        obsd_t *o = &obs.data[i];

        int prn = 0;
        int sys = satsys(o->sat, &prn);   // RTKLIB: 返回 SYS_GPS / SYS_GLO / SYS_GAL / SYS_QZS / SYS_CMP / ...
        if ( (sys & (SYS_GPS | SYS_CMP)) == 0 ) {
            continue; // 非 GPS/BDS 直接跳过
        }

        char tbuf[64]; time_to_str(o->time, tbuf, sizeof tbuf);
        char satid[8]; satno2id(o->sat, satid);

        // 解析年用于路径
        double ep[6]; time2epoch(o->time, ep);
        const int year = (int)ep[0];

        // 测站坐标（直接加入每行，表头已经有 rec_x/y/z）
        double rec_x = sta.pos[0];
        double rec_y = sta.pos[1];
        double rec_z = sta.pos[2];

        // 为该历元/卫星收集所有目标信号的 C/N0
        // 初始化为 NaN（输出为空格即可）；我们直接用标记位来控制输出为空
        std::vector<double> cn0(expected_codes_vec.size(), -1.0);
        std::vector<bool>   has(expected_codes_vec.size(), false);

        // RTKLIB 每条观测有多频槽，逐槽填入
        for (int j=0;j<NFREQ+NEXOBS;++j){
            if (o->SNR_FIELD[j] <= 0 || o->code[j] == CODE_NONE) continue;
            const unsigned char code = o->code[j];
            if (!expected_codes.count(code)) continue;

            // 记 seen
            seen_by_sat[satid].insert(code);

            // 写入对应列
            auto it = std::find(expected_codes_vec.begin(), expected_codes_vec.end(), code);
            if (it != expected_codes_vec.end()) {
                size_t col = (size_t)std::distance(expected_codes_vec.begin(), it);
                cn0[col] = o->SNR_FIELD[j];
                has[col] = true;
            }
        }

        // 若至少有一个目标信号出现，则输出一行
        bool any = std::any_of(has.begin(), has.end(), [](bool v){return v;});
        if (!any) continue;

        FILE* sink = ensure_sink(year);
        if (!sink) continue;

        // 固定列
        std::fprintf(sink, "%s,%s,%s,%.4f,%.4f,%.4f",
                     tbuf, satid, station_tag.c_str(), rec_x, rec_y, rec_z);

        // 各信号列：有值写数值；无值留空
        for (size_t k=0;k<expected_codes_vec.size();++k){
            if (has[k]) std::fprintf(sink, ",%.2f", cn0[k]);
            else        std::fprintf(sink, ",");
        }
        std::fprintf(sink, "\n");
    }

    // 缺失信号提示
    for (const auto& kv : seen_by_sat){
        const auto& sat = kv.first;
        for (auto code_needed : expected_codes_vec){
            if (!kv.second.count(code_needed)){
                auto it = code2label.find(code_needed);
                const std::string sig = (it==code2label.end()) ? "S??" : it->second;
                LOG_WARN("MISSING SIGNAL: sat=%s signal=%s file=%s",
                         sat.c_str(), sig.c_str(), rnx_path.string().c_str());
            }
        }
    }

    if (fo) std::fclose(fo);

    std::free(obs.data);
    std::free(nav.eph);
    std::free(nav.geph);
    std::free(nav.seph);
}
