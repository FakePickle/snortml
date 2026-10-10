# SnortML+ Inference Daemon

Network anomaly detection calibrated on your own traffic. It learns what
normal looks like on your network from a recording of that network, then
flags flows and hosts that are rare by that standard. It can capture
traffic itself, or sit behind Suricata, Zeek or Snort and report only what
their signatures did not.

No TensorFlow and no ONNX runtime: scoring is plain numpy. Runs on Linux,
Raspberry Pi (64-bit) and Windows.

---

## How it decides

1. **Same features for learning and scoring.** The baseline is built from
   the same flow records the daemon scores, so a feature never means one
   thing in training and another live.
2. **Each flow is compared with its own kind.** A ping is scored against
   other pings and an HTTPS flow against other HTTPS flows (`tcp/443`,
   `udp/53`, `icmp`, ...). Being a minority protocol is not an anomaly.
3. **Two layers.** The *flow* layer looks at the shape of one connection
   (size, duration, timing, packet sizes). The *host* layer looks at what one
   host did in a minute (flows, distinct destinations and ports, unanswered
   targets, data sent), which is where scans, floods and sweeps show up.
4. **A score is a rarity, and the threshold is an alert budget.** Every
   score is converted to "1 in N flows like the baseline would look this
   unusual". You say how many alerts a day you want; the threshold follows
   from the baseline.
5. **Incidents, not flows.** Repeats of the same incident are counted, not
   re-sent. A one-hour transfer is one alert, not sixty.
6. **New services.** The first use of a well-known port that never appeared
   in the baseline (first SSH, first SMB, first telnet) is reported.

The model per context is a rank-Gaussian transform of each feature, a
shrinkage Mahalanobis distance, and a small autoencoder's reconstruction
error, calibrated on held-out baseline traffic with an extreme-value tail.

---

## Install

```bash
# Linux / Raspberry Pi OS (64-bit)
sudo apt install libpcap-dev
python3 -m venv .venv && . .venv/bin/activate
pip install .

# or, with uv
uv sync
```

Windows: install [Npcap](https://npcap.com/#download) with "WinPcap API
compatibility" checked, then `pip install .` in an Administrator terminal.

Python 3.10 or newer. 32-bit ARM is not supported (nfstream has no wheel).

---

## Quick start

```bash
# 1. Find your interface
python -m snortml_inference interfaces

# 2. Record normal traffic and fit a baseline from it.
#    A day is the minimum worth having; a week is better.
sudo python -m snortml_inference baseline --source eth0 --duration 86400

# 3. Run
sudo python -m snortml_inference run --interface eth0
```

Put the interface in `config.yaml` (`source.interface`) and step 3 is just
`python -m snortml_inference run`. On Windows the interface is the
`\Device\NPF_{...}` name that the `interfaces` command prints.

Try it without capturing anything:

```bash
python -m snortml_inference baseline --source monday.pcap
python -m snortml_inference replay   --source tuesday.pcap
```

`replay` runs a recorded capture or log through exactly the code the daemon
runs and prints what would have alerted.

| Command | What it does |
|---|---|
| `baseline --source X` | Record from an interface or read a file, then fit the baseline |
| `fit` | Re-fit the baseline from flows the daemon has recorded (`data/`) |
| `run` | Score live traffic or a growing log file |
| `replay --source FILE` | Score a pcap, conn.log, eve.json or flow CSV offline |
| `info` | Describe the current baseline and its thresholds |
| `interfaces` | List capture interfaces |

---

## Behind an existing IDS

This does not replace Snort, Suricata or Zeek. The signature IDS stays the
first layer and owns everything its rules catch. This is the second layer:
it learns what normal looks like from the traffic the rules let through,
and reports what is unusual in that remainder.

A connection the IDS alerted on is therefore left alone here. It is not
scored, not counted in the per-host windows, not reported as a new service,
and not learned as part of the baseline. Set `ids.on_match: tag` to score
such flows anyway and mark the alerts.

**Suricata.** Read its flow records instead of capturing a second time.
Flows Suricata already alerted on are left to Suricata.

```yaml
source:
  type: suricata
  path: /var/log/suricata/eve.json
```

**Zeek.** Follow `conn.log` (TSV or JSON).

```yaml
source:
  type: zeek
  path: /opt/zeek/logs/current/conn.log
```

**Snort 3, or Suricata while capturing yourself.** Keep `type: live` and
point the daemon at the IDS's alert file; connections it alerted on are
then skipped (`on_match: suppress`) or marked (`on_match: tag`).

```yaml
ids:
  snort_json: /var/log/snort/alert_json.txt
  # suricata_eve: /var/log/suricata/eve.json
```

Build the baseline from the same kind of input you will run on
(`baseline --source /var/log/suricata/eve.json`), with the same `ids`
settings, so the alert file is followed while the baseline is recorded. Zeek and Suricata report
no per-packet statistics, so those baselines use the size, duration and
state features only.

---

## Alerts

```
[2026-10-02 14:03:53] [MEDIUM] flow anomaly (upload) | 10.0.2.15:51515 -> 203.0.113.50:443 tcp/443 | 1 in 362,300 | bytes to server 57.2 MB (usually 146 B to 5.5 KB); duration 60.0 s (usually 193 ms to 21.6 s)
[2026-10-02 14:17:00] [CRITICAL] host anomaly (fan-out) | 10.0.2.15 | 1 in 13,500 | distinct destination ports 231 (usually 1 to 4)
[2026-10-02 14:52:10] [MEDIUM] new service | 10.0.2.15 -> 198.51.100.22:22 tcp/22 | tcp/22 was never used in the baseline (44,466 flows); 200 connections in the last hour
```

Each alert says how rare the event is and which measurements made it so,
with the usual range for that context. Most also carry `resembles`: the
behaviour the pattern usually turns out to be (port scan, sweep of many
hosts, flood or brute force, possible data exfiltration, tunnel). That is
a lead for whoever triages the alert. It is not an identification: there is
no signature and no label for something never seen before, only a
description of how it differs from normal.

`alerts/alerts.jsonl` has one JSON object per alert with Suricata-style
field names (`src_ip`, `dest_ip`, `dest_port`, `proto`), plus `layer`,
`kind`, `severity`, `p_value`, `one_in`, `reasons` and `detail`.

Severity is how far past the alert threshold the event is, one step per
factor of ten: LOW, MEDIUM, HIGH, CRITICAL.

Kinds: `upload`, `download`, `internal-download`, `volume`, `timing`,
`packet-size`, `connection-state`, `direction` (flow layer); `fan-out`,
`unanswered`, `upload` (host layer); `new-service`.

---

## Tuning

Everything is in `config.yaml`; the comments there explain each setting.
The ones that matter most:

- `scoring.alerts_per_day`: the alert budget. This replaces the old score
  threshold.
- `scoring.drift_margin`: how much stricter than the baseline alone
  suggests. Raise it if new days are noisy; lower it for more sensitivity.
- `alerts.mute`: anomaly kinds not to alert on. Large downloads from the
  internet are muted by default; set `mute: []` to see them.
- `novelty.allow`: services to accept, e.g. `["tcp/22"]`.
- `home_networks`: your public IPv6 prefix, if your devices have global
  addresses. Private ranges are known already.

`python -m snortml_inference info` shows the thresholds the current
settings give, and any context that has too little baseline to alert yet.

## Keeping the baseline fresh

Networks change. The daemon keeps the flows it sees in `data/` (30 days by
default), and

```bash
python -m snortml_inference fit --days 14
```

re-fits the baseline from them. A running daemon notices the new baseline
within a minute and reloads it. A weekly cron job is a reasonable start:

```
0 4 * * 1  cd /opt/snortml && .venv/bin/python -m snortml_inference fit --days 14
```

If the daemon logs that far more flows cross the threshold than expected,
or the hourly alert cap is hit, the baseline is out of date.

Whatever is in the recording becomes normal, including an attack that was
running at the time. Record the baseline when you believe the network is
clean.

---

## Run as a service (Linux / Raspberry Pi)

```ini
# /etc/systemd/system/snortml.service
[Unit]
Description=SnortML+ anomaly daemon
After=network-online.target

[Service]
WorkingDirectory=/opt/snortml
ExecStart=/opt/snortml/.venv/bin/python -m snortml_inference run
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now snortml
sudo journalctl -u snortml -f
```

Following a Zeek or Suricata log needs no packet-capture privileges, only
read access to the log.

## Sending alerts to Wazuh

Wazuh decodes JSON lines on its own. This example has not been tested
against a live Wazuh.

```xml
<!-- agent ossec.conf -->
<localfile>
  <log_format>json</log_format>
  <location>/opt/snortml/alerts/alerts.jsonl</location>
</localfile>

<!-- manager local_rules.xml -->
<group name="anomaly,">
  <rule id="100900" level="5">
    <decoded_as>json</decoded_as>
    <field name="event_type">^anomaly$</field>
    <description>Network anomaly ($(layer)/$(kind)) from $(src_ip)</description>
  </rule>
  <rule id="100901" level="10">
    <if_sid>100900</if_sid>
    <field name="severity">^HIGH$|^CRITICAL$</field>
    <description>Network anomaly, $(severity): $(layer)/$(kind) from $(src_ip)</description>
  </rule>
</group>
```

A webhook (`alerts.webhook`) posts each alert as JSON, or as plain text for
services such as ntfy.

---

## What it catches, and what it does not

Measured on 2 October 2026 with the default configuration, on public
captures of one Windows machine's normal use over four days (Stratosphere
CTU-Normal-23, 24, 25 and 20).

**Normal traffic.** Baseline from two days (6.7 hours of traffic). Two
unseen days, 4.7 hours in total: **1 alert**, severity LOW.

**Attacks.** Synthetic attack flows mixed into an unseen day, one type at a
time:

| Attack | Result | Raised by |
|---|---|---|
| 600 MB upload over HTTPS | caught | flow and host (upload) |
| 20 MB upload over HTTPS | caught | flow and host (upload) |
| Port scan, 1,000 ports in 30 s | caught, CRITICAL | host (fan-out), new service |
| Slow scan, 60 ports a minute | caught, HIGH | host (fan-out), new service |
| SMB sweep of a /24 | caught, HIGH | host (unanswered), new service |
| SYN flood, 20,000 a minute | caught, LOW | host (fan-out) |
| Telnet scan, 3,000 hosts a minute | caught, CRITICAL | host (unanswered), new service |
| SSH brute force, 200 tries | caught, MEDIUM | new service |
| DNS tunnel | missed | |
| ICMP tunnel | missed | |
| Command-and-control beacon, one small request a minute | missed | |

Read these numbers with care:

- The attack flows are synthetic, built to look like what nfstream reports
  for such traffic. They are not captures of real tools.
- One machine and 6.7 hours is a small baseline. Rare contexts (anything
  that is not web traffic on that machine) had too little data to alert,
  which is why the two tunnels were missed.
- Low-and-slow traffic that looks like ordinary browsing is out of reach
  for flow statistics. Catching beacons needs timing analysis across flows,
  which this does not do.
- Something that is rare is not necessarily malicious, and the reverse.

The live daemon was tested end to end by following a growing Suricata log
in a real process, and on pcap files through nfstream. Capturing from a
live interface runs the same code but was not exercised in the test
environment (no capture privileges there), and neither was Windows.

---

## Layout

```
snortml_inference/
  flows.py      the flow record every source is normalized to; CSV storage
  sources.py    nfstream (interface / pcap), Zeek conn.log, Suricata eve.json
  features.py   orientation, context groups, model inputs
  windows.py    per-host one-minute windows
  model.py      the detector: transform, distance, autoencoder, calibration
  baseline.py   fit / save / load a baseline; alert budget -> threshold
  scorer.py     scoring and explanations
  alerter.py    incidents, muting, rate cap, outputs
  ids.py        which connections a signature IDS already alerted on
  daemon.py     the engine and the live loop
  config.py     defaults and config loading
tests/          pytest suite (python -m pytest)
legacy/         the first version of the daemon and its model, for reference
```

A baseline is a directory with `manifest.json` and `arrays.npz`. Neither is
a pickle. The baseline, the recorded flows, the alerts and the logs all
contain your network's addresses; `.gitignore` keeps them out of the
repository.
