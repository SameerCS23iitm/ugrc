#!/bin/bash

echo "Extracting Modbus packets..."

# -----------------------
# Benign Network-Wide
# -----------------------
tshark -r benign-merged-nw.pcap \
-Y "modbus" \
-T fields \
-e frame.time_epoch \
-e ip.src \
-e ip.dst \
-e tcp.srcport \
-e tcp.dstport \
-e frame.len \
-e tcp.flags \
-e modbus.func_code \
-e modbus.reference_num \
-e modbus.word_cnt \
-e modbus.exception_code \
-E header=y \
-E separator=, \
-E occurrence=f \
> benign_nw.csv


# -----------------------
# Compromised SCADA Attack (Substation-Wide)
# -----------------------
tshark -r comp-scada-attack-ssw.pcap \
-Y "modbus" \
-T fields \
-e frame.time_epoch \
-e ip.src \
-e ip.dst \
-e tcp.srcport \
-e tcp.dstport \
-e frame.len \
-e tcp.flags \
-e modbus.func_code \
-e modbus.reference_num \
-e modbus.word_cnt \
-e modbus.exception_code \
-E header=y \
-E separator=, \
-E occurrence=f \
> cscada_attack_ssw.csv


# -----------------------
# External Attack Network-Wide
# -----------------------
tshark -r external-attack-nw.pcap \
-Y "modbus" \
-T fields \
-e frame.time_epoch \
-e ip.src \
-e ip.dst \
-e tcp.srcport \
-e tcp.dstport \
-e frame.len \
-e tcp.flags \
-e modbus.func_code \
-e modbus.reference_num \
-e modbus.word_cnt \
-e modbus.exception_code \
-E header=y \
-E separator=, \
-E occurrence=f \
> ext_attack_nw.csv


echo "Extraction complete."