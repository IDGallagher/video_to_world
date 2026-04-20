#include "GDeflate.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

bool ReadFileBytes(const char* path, std::vector<std::uint8_t>& out_bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        return false;
    }

    const std::streamsize size = input.tellg();
    if (size < 0)
    {
        return false;
    }

    out_bytes.resize(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    return input.read(reinterpret_cast<char*>(out_bytes.data()), size).good();
}

bool WriteFileBytes(const char* path, const std::vector<std::uint8_t>& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        return false;
    }

    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

int CompressFile(const char* input_path, const char* output_path, std::uint32_t compression_level)
{
    std::vector<std::uint8_t> input_bytes;
    if (!ReadFileBytes(input_path, input_bytes))
    {
        std::cerr << "Failed to read input file: " << input_path << std::endl;
        return 1;
    }

    const std::size_t bound = GDeflate::CompressBound(input_bytes.size());
    std::vector<std::uint8_t> output_bytes(bound);
    std::size_t output_size = bound;
    if (!GDeflate::Compress(
            output_bytes.data(),
            &output_size,
            input_bytes.data(),
            input_bytes.size(),
            compression_level,
            0))
    {
        std::cerr << "GDeflate compression failed for: " << input_path << std::endl;
        return 1;
    }

    output_bytes.resize(output_size);
    if (!WriteFileBytes(output_path, output_bytes))
    {
        std::cerr << "Failed to write output file: " << output_path << std::endl;
        return 1;
    }

    return 0;
}

int DecompressFile(const char* input_path, const char* output_path, std::size_t output_size)
{
    std::vector<std::uint8_t> input_bytes;
    if (!ReadFileBytes(input_path, input_bytes))
    {
        std::cerr << "Failed to read input file: " << input_path << std::endl;
        return 1;
    }

    std::vector<std::uint8_t> output_bytes(output_size);
    if (!GDeflate::Decompress(
            output_bytes.data(),
            output_bytes.size(),
            input_bytes.data(),
            input_bytes.size(),
            0))
    {
        std::cerr << "GDeflate decompression failed for: " << input_path << std::endl;
        return 1;
    }

    if (!WriteFileBytes(output_path, output_bytes))
    {
        std::cerr << "Failed to write output file: " << output_path << std::endl;
        return 1;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: divstream_gdeflate_cli compress <input_file> <output_file> <level>\n"
                     "   or: divstream_gdeflate_cli decompress <input_file> <output_file> <output_size>" << std::endl;
        return 1;
    }

    const std::string command = argv[1];
    if (command == "compress")
    {
        if (argc != 5)
        {
            std::cerr << "Usage: divstream_gdeflate_cli compress <input_file> <output_file> <level>" << std::endl;
            return 1;
        }

        const long level = std::strtol(argv[4], nullptr, 10);
        if (level < static_cast<long>(GDeflate::MinimumCompressionLevel) ||
            level > static_cast<long>(GDeflate::MaximumCompressionLevel))
        {
            std::cerr << "Compression level must be between " << GDeflate::MinimumCompressionLevel
                      << " and " << GDeflate::MaximumCompressionLevel << std::endl;
            return 1;
        }

        return CompressFile(argv[2], argv[3], static_cast<std::uint32_t>(level));
    }

    if (command == "decompress")
    {
        if (argc != 5)
        {
            std::cerr << "Usage: divstream_gdeflate_cli decompress <input_file> <output_file> <output_size>" << std::endl;
            return 1;
        }

        const unsigned long long output_size = std::strtoull(argv[4], nullptr, 10);
        if (output_size == 0)
        {
            std::cerr << "Output size must be greater than zero." << std::endl;
            return 1;
        }

        return DecompressFile(argv[2], argv[3], static_cast<std::size_t>(output_size));
    }

    std::cerr << "Unsupported command: " << command << std::endl;
    return 1;
}
