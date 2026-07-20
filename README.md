# IQ Option MCP Server

An MCP server that lets Claude read your IQ Option account, pull market data,
and (optionally) place trades. Built on the community
[iqoptionapi](https://github.com/iqoptionapi/iqoptionapi) websocket library.

> **Unofficial API warning:** IQ Option has no public API. This library is
> reverse-engineered — logins can break without notice when IQ Option changes
> their backend, automated trading may violate their Terms of Service, and
> accounts using it can in principle be flagged. Use at your own risk.

## Setup

```bash
cd ~/Desktop/Developer/iqoption
python3 -m venv .venv
.venv/bin/pip install mcp ./vendor/iqoptionapi
cp .env.example .env   # then fill in IQ_EMAIL / IQ_PASSWORD
```

Register with Claude Code (already done on this machine):

```bash
claude mcp add --scope user iqoption -- \
  ~/Desktop/Developer/iqoption/.venv/bin/python \
  ~/Desktop/Developer/iqoption/server.py
```

## Safety model

- Connects to the **PRACTICE** (demo) balance by default.
- Every trading tool (`iq_place_binary`, `iq_place_order`, `iq_close_position`,
  `iq_cancel_order`) **refuses to act on the REAL balance** unless you set
  `IQ_ALLOW_REAL=1` in `.env`. Viewing REAL balances/positions is always fine.
- Accounts with SMS two-factor auth: the first `iq_connect` call triggers the
  SMS; call `iq_connect` again with `sms_code` to finish logging in.

## Tools

| Tool | What it does |
|---|---|
| `iq_connect` | Connect / reconnect (handles SMS 2FA) |
| `iq_status` | Connection, balance mode, balance, currency |
| `iq_switch_balance` | Switch PRACTICE ↔ REAL |
| `iq_reset_practice_balance` | Refill the demo balance |
| `iq_find_asset` | Search asset names (EURUSD, APPLE, EURUSD-OTC…) |
| `iq_get_candles` | Historical OHLC candles |
| `iq_open_assets` | What's currently open for trading, per market |
| `iq_payouts` | Binary/turbo payout ratios per asset |
| `iq_instruments` | Tradable CFD/forex/crypto instrument ids |
| `iq_positions` | Open positions per instrument type |
| `iq_place_binary` | Binary option trade (call/put) |
| `iq_binary_result` | Wait for a binary option's win/lose outcome |
| `iq_place_order` | Margin order (CFD/forex/crypto) with TP/SL |
| `iq_close_position` | Close an open margin position |
| `iq_cancel_order` | Cancel a pending order |

## Notes

- The vendored library lives in `vendor/iqoptionapi` (gitignored; re-clone from
  GitHub if missing).
- Library calls that lose their websocket reply would busy-wait forever; the
  server wraps every call in a timeout so a stuck call errors out instead of
  hanging Claude.
