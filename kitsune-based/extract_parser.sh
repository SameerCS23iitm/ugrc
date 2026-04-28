#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_PY="$SCRIPT_DIR/modbus_parser.py"
PCAP_DIR="$SCRIPT_DIR/../data/attack/compromised-scada/substation-wide-capture"
CSV_OUT_DIR="$SCRIPT_DIR/../train/extracted"
PORT="502"
LABEL=""

if [[ ! -f "$PARSER_PY" ]]; then
	echo "Error: parser file not found: $PARSER_PY"
	exit 1
fi

if [[ ! -d "$PCAP_DIR" ]]; then
	echo "Error: PCAP directory not found: $PCAP_DIR"
	exit 1
fi

mkdir -p "$CSV_OUT_DIR"

count=0
ok=0
fail=0

while IFS= read -r -d '' pcap_file; do
	count=$((count + 1))

	base_name="$(basename "$pcap_file")"
	stem="${base_name%.pcap}"
	stem="${stem%.pcapng}"
	out_csv="$CSV_OUT_DIR/${stem}.csv"

	echo "[$count] Processing: $pcap_file"
	if [[ -n "$LABEL" ]]; then
		if python3 "$PARSER_PY" --pcap "$pcap_file" --out "$out_csv" --label "$LABEL" --port "$PORT"; then
			ok=$((ok + 1))
		else
			echo "  Failed: $pcap_file"
			fail=$((fail + 1))
		fi
	else
		if python3 "$PARSER_PY" --pcap "$pcap_file" --out "$out_csv" --port "$PORT"; then
			ok=$((ok + 1))
		else
			echo "  Failed: $pcap_file"
			fail=$((fail + 1))
		fi
	fi
done < <(find "$PCAP_DIR" -type f \( -iname "*.pcap" -o -iname "*.pcapng" \) -print0 | sort -z)

echo
echo "Done."
echo "Total files found : $count"
echo "Successful CSVs   : $ok"
echo "Failed files      : $fail"
echo "CSV output dir    : $CSV_OUT_DIR"