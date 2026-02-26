# trim_tokens.py
"""
Trim agathyam_tokens.txt to the top N tokens (default: 2000).
Usage: python trim_tokens.py [N]
"""
import sys

input_file = "agathyam_tokens.txt"
output_file = "agathyam_tokens_trimmed.txt"
def main():
    N = 2000
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    trimmed = lines[:N]
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(trimmed)
    print(f"Wrote top {N} tokens to {output_file}")

if __name__ == "__main__":
    main()
