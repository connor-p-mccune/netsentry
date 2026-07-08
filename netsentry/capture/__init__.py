"""Packet-capture ingestion: raw PCAP -> CIC-schema flow features.

Turns NetSentry from a consumer of pre-computed flow features into a system
that can score an actual packet capture: a pure-stdlib libpcap reader
(:mod:`netsentry.capture.pcap`), a bidirectional flow assembler that computes
the full CICFlowMeter feature schema (:mod:`netsentry.capture.flows`), and a
synthetic capture writer for demos and tests (:mod:`netsentry.capture.demo`).
"""

from netsentry.capture.demo import write_demo_pcap
from netsentry.capture.flows import FlowAssembler, extract_flows
from netsentry.capture.pcap import PacketRecord, PcapReadError, PcapStats, read_pcap
