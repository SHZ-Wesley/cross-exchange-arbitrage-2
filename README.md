# Cross-Exchange Arbitrage Bot (跨交易所套利机器人)

这是一个跨交易所套利机器人，旨在捕获 **Lighter.xyz** (作为 Taker 端) 与其他永续合约交易所 (作为 Maker 端) 之间的价差。

目前支持的 Maker 端交易所包括：

- **EdgeX**
- **Paradex**
- **GRVT**

## 🚀 功能特性

- **Maker-Taker 策略**：在 Maker 交易所（如 Paradex/GRVT）挂出 Post-Only 限价单，成交后立即在 Lighter 上吃单对冲。
- **多交易所支持**：通过统一的接口支持多种 Perpdex，方便扩展。
- **自动对冲与持仓平衡**：跟踪两边交易所的持仓，并在达到阈值时停止开仓。
- **WebSocket 集成**：实时监听订单簿数据和订单状态更新，确保极低延迟。
- **灵活配置**：支持通过命令行参数调整交易对、下单数量、价差阈值等。

## 🛠️ 安装指南

### 1. 克隆仓库

```
git clone [https://github.com/your-username/cross-exchange-arbitrage.git](https://github.com/your-username/cross-exchange-arbitrage.git)
cd cross-exchange-arbitrage
```

### 2. 安装依赖

除了基础依赖外，你还需要安装特定交易所的 SDK。

**基础依赖：**

```
pip install -r requirements.txt
```

**交易所 SDK (必须安装)：**

Paradex SDK (需从 GitHub 安装最新版):

```
pip install git+[https://github.com/tradeparadex/paradex-py.git](https://github.com/tradeparadex/paradex-py.git)
```

GRVT SDK:

```
pip install grvt-pysdk
```

## ⚙️ 配置说明

在项目根目录下创建一个 `.env` 文件，并填入以下配置信息。请根据你使用的交易所填写相应部分。

```
# ==========================================
# Lighter 配置 (Taker 端 - 必须配置)
# ==========================================
# 你的 Lighter 账户私钥 (通常与以太坊私钥相同)
API_KEY_PRIVATE_KEY=0x...
# Lighter 账户索引 (默认为 0)
LIGHTER_ACCOUNT_INDEX=0
# Lighter API Key 索引 (默认为 0)
LIGHTER_API_KEY_INDEX=0

# ==========================================
# EdgeX 配置 (如果你使用 EdgeX)
# ==========================================
EDGEX_WALLET_ADDRESS=0x...
EDGEX_PRIVATE_KEY=0x...
# 账户名称 (可选，用于日志区分)
ACCOUNT_NAME=MainAccount

# ==========================================
# Paradex 配置 (如果你使用 Paradex)
# ==========================================
# 以太坊 L1 地址
PARADEX_L1_ADDRESS=0x...
# Paradex L2 私钥 (从 Paradex 网页端导出，不是助记词)
PARADEX_L2_PRIVATE_KEY=0x...
# Paradex L2 地址 (可选，程序会自动推导)
PARADEX_L2_ADDRESS=0x...
# 环境: prod (生产) 或 testnet (测试网)
PARADEX_ENVIRONMENT=prod

# ==========================================
# GRVT 配置 (如果你使用 GRVT)
# ==========================================
# GRVT 账户 ID
GRVT_TRADING_ACCOUNT_ID=...
# GRVT API Key
GRVT_API_KEY=...
# GRVT 私钥
GRVT_PRIVATE_KEY=...
# 环境: prod (生产) 或 testnet (测试网)
GRVT_ENVIRONMENT=prod
```

## 🏃‍♂️ 运行指南

使用 `arbitrage.py` 作为入口文件启动机器人。

### 基本用法

```
python arbitrage.py --maker <交易所名称> --ticker <币种> --size <数量>
```

### 参数说明

| 参数                | 说明                                             | 默认值     |
| ------------------- | ------------------------------------------------ | ---------- |
| `--maker`           | 选择 Maker 端交易所 (`edgex`, `paradex`, `grvt`) | `edgex`    |
| `--ticker`          | 交易对代码 (如 `BTC`, `ETH`)                     | `BTC`      |
| `--size`            | 单笔订单数量 (必须指定)                          | 无         |
| `--fill-timeout`    | Maker 订单等待成交超时时间 (秒)                  | 5          |
| `--max-position`    | 最大单边持仓限制                                 | 0 (不限制) |
| `--long-threshold`  | 做多触发价差阈值 (Lighter Bid - Maker Bid)       | 10         |
| `--short-threshold` | 做空触发价差阈值 (Maker Ask - Lighter Ask)       | 10         |

### 运行示例

**1. 在 Paradex 上套利 BTC (每次 0.001 BTC):**

```
python arbitrage.py --maker paradex --ticker BTC --size 0.001
```

**2. 在 GRVT 上套利 ETH (每次 0.01 ETH):**

```
python arbitrage.py --maker grvt --ticker ETH --size 0.01
```

**3. 在 EdgeX 上套利 SOL (每次 1 SOL)，设置最大持仓为 10 SOL:**

```
python arbitrage.py --maker edgex --ticker SOL --size 1 --max-position 10
```

## 📈 策略逻辑

1. **监听价差**：程序同时监听 Maker 交易所 (如 Paradex) 和 Taker 交易所 (Lighter) 的订单簿 (BBO)。
2. **触发条件**：
   - **做多逻辑**：当 `Lighter Bid Price` - `Maker Bid Price` > `long_threshold` 时，认为 Maker 端价格偏低。
   - **做空逻辑**：当 `Maker Ask Price` - `Lighter Ask Price` > `short_threshold` 时，认为 Maker 端价格偏高。
3. **执行套利**： 限价单。
   - 一旦 Maker 单成交，程序立即在 Lighter 发送市价单 (Market Order) 进行对冲。
4. **循环**：重复上述过程，赚取 Maker 返佣和价差。

## ⚠️ 免责声明

本软件仅供教育和研究目的使用。加密货币交易具有极高的风险，可能导致资金损失。开发者不对因使用本软件而产生的任何损失负责。使用前请确保您了解相关风险并已配置正确的 API 权限。