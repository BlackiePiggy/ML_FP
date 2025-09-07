#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include "config.h"

// 从 RINEX 读取并按 年/测站 输出单表 CSV；
// CSV 内包含所有期望信号（每个信号一列）。
void extract_rnx_to_csvs(const std::filesystem::path& rnx_path,
                         const std::filesystem::path& out_root,
                         const std::string& station_tag,
                         const std::vector<unsigned char>& expected_codes,
                         const std::unordered_map<unsigned char, std::string>& code2label);
