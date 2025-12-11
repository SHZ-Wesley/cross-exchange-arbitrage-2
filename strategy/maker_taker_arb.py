"""Main arbitrage trading bot for Maker (Generic) and Lighter exchanges."""
import asyncio
import signal
import logging
import os
import sys
import time
import requests
import traceback
from decimal import Decimal
from typing import Tuple, Dict, Any

from lighter.signer_client import SignerClient
from exchanges.factory import ExchangeFactory

from .data_logger import DataLogger
from .order_book_manager import OrderBookManager
from .websocket_manager import WebSocketManagerWrapper
from .order_manager import OrderManager
from .position_tracker import PositionTracker


class Config:
    """Simple config class to wrap dictionary for clients if needed."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class MakerTakerArb:
    """Arbitrage trading bot: makes post-only orders on Maker Exchange, and market orders on Lighter (Taker)."""

    def __init__(self, maker_exchange: str, ticker: str, order_quantity: Decimal,
                 fill_timeout: int = 5, max_position: Decimal = Decimal('0'),
                 long_ex_threshold: Decimal = Decimal('10'),
                 short_ex_threshold: Decimal = Decimal('10')):
        """Initialize the arbitrage trading bot."""
        self.maker_exchange_name = maker_exchange
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.max_position = max_position
        self.stop_flag = False
        self._cleanup_done = False

        self.long_ex_threshold = long_ex_threshold
        self.short_ex_threshold = short_ex_threshold

        # Setup logger
        self._setup_logger()

        # Initialize modules
        self.data_logger = DataLogger(exchange=maker_exchange, ticker=ticker, logger=self.logger)
        self.order_book_manager = OrderBookManager(self.logger)
        self.ws_manager = WebSocketManagerWrapper(self.order_book_manager, self.logger)
        self.order_manager = OrderManager(self.order_book_manager, self.logger)

        # Initialize clients (will be set later)
        self.maker_client = None
        self.lighter_client = None

        # Configuration
        self.lighter_base_url = "https://mainnet.zklighter.elliot.ai"
        self.account_index = int(os.getenv('LIGHTER_ACCOUNT_INDEX'))
        self.api_key_index = int(os.getenv('LIGHTER_API_KEY_INDEX'))
        
        # Contract/market info (will be set during initialization)
        self.maker_contract_id = None
        self.maker_tick_size = None
        self.lighter_market_index = None
        self.base_amount_multiplier = None
        self.price_multiplier = None
        self.tick_size = None

        # Position tracker (will be initialized after clients)
        self.position_tracker = None

        # Setup callbacks
        self._setup_callbacks()

    def _setup_logger(self):
        """Setup logging configuration."""
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/{self.maker_exchange_name}_{self.ticker}_log.txt"

        self.logger = logging.getLogger(f"arbitrage_bot_{self.ticker}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Disable verbose logging from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('websockets').setLevel(logging.WARNING)

        # Create file handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def _setup_callbacks(self):
        """Setup callback functions for order updates."""
        self.ws_manager.set_callbacks(
            on_lighter_order_filled=self._handle_lighter_order_filled,
            # Note: on_edgex_order_update removed from ws_manager, handled via client callback
        )
        self.order_manager.set_callbacks(
            on_order_filled=self._handle_lighter_order_filled
        )

    def _handle_lighter_order_filled(self, order_data: dict):
        """Handle Lighter order fill."""
        try:
            order_data["avg_filled_price"] = (
                Decimal(order_data["filled_quote_amount"]) /
                Decimal(order_data["filled_base_amount"])
            )
            if order_data["is_ask"]:
                order_data["side"] = "SHORT"
                order_type = "OPEN"
                if self.position_tracker:
                    self.position_tracker.update_lighter_position(
                        -Decimal(order_data["filled_base_amount"]))
            else:
                order_data["side"] = "LONG"
                order_type = "CLOSE"
                if self.position_tracker:
                    self.position_tracker.update_lighter_position(
                        Decimal(order_data["filled_base_amount"]))

            client_order_index = order_data["client_order_id"]
            self.logger.info(
                f"[{client_order_index}] [{order_type}] [Lighter] [FILLED]: "
                f"{order_data['filled_base_amount']} @ {order_data['avg_filled_price']}")

            # Log trade to CSV
            self.data_logger.log_trade_to_csv(
                exchange='lighter',
                side=order_data['side'],
                price=str(order_data['avg_filled_price']),
                quantity=str(order_data['filled_base_amount'])
            )

            # Mark execution as complete
            self.order_manager.lighter_order_filled = True
            self.order_manager.order_execution_complete = True

        except Exception as e:
            self.logger.error(f"Error handling Lighter order result: {e}")

    def _handle_maker_order_update(self, order_info: dict):
        """Handle Maker order update from WebSocket/Callback."""
        try:
            # Normalized order info dict
            contract_id = order_info.get('contract_id')
            if contract_id != self.maker_contract_id:
                return

            # Check if this is the order we placed?
            # OrderManager handles tracking, here we just process fills for logging/hedging
            
            order_id = order_info.get('order_id')
            status = order_info.get('status')
            side = order_info.get('side', '').lower()
            filled_size = Decimal(str(order_info.get('filled_size', '0')))
            size = Decimal(str(order_info.get('size', '0')))
            price = order_info.get('price', '0')

            if side == 'buy':
                order_type = "OPEN"
            else:
                order_type = "CLOSE"

            if status == 'CANCELED' and filled_size > 0:
                status = 'FILLED'

            # Pass update to order manager to handle logic state
            self.order_manager.handle_maker_order_update({
                'order_id': order_id,
                'side': side,
                'status': status,
                'size': size,
                'price': price,
                'contract_id': contract_id,
                'filled_size': filled_size
            })

            # Handle filled orders logic
            if status == 'FILLED' and filled_size > 0:
                if side == 'buy':
                    if self.position_tracker:
                        self.position_tracker.update_edgex_position(filled_size) # reusing variable name for maker position
                else:
                    if self.position_tracker:
                        self.position_tracker.update_edgex_position(-filled_size)

                self.logger.info(
                    f"[{order_id}] [{order_type}] [{self.maker_exchange_name}] [{status}]: {filled_size} @ {price}")

                if filled_size > 0.0001:
                    # Log Maker trade to CSV
                    self.data_logger.log_trade_to_csv(
                        exchange=self.maker_exchange_name,
                        side=side,
                        price=str(price),
                        quantity=str(filled_size)
                    )

            elif status != 'FILLED':
                if status == 'OPEN':
                    self.logger.info(f"[{order_id}] [{order_type}] [{self.maker_exchange_name}] [{status}]: {size} @ {price}")
                else:
                    self.logger.info(
                        f"[{order_id}] [{order_type}] [{self.maker_exchange_name}] [{status}]: {filled_size} @ {price}")

        except Exception as e:
            self.logger.error(f"Error handling Maker order update: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        # Prevent multiple shutdown calls
        if self.stop_flag:
            return

        self.stop_flag = True

        if signum is not None:
            self.logger.info("\n🛑 Stopping...")
        else:
            self.logger.info("🛑 Stopping...")

        # Shutdown WebSocket connections
        try:
            if self.ws_manager:
                self.ws_manager.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down WebSocket manager: {e}")

        # Close data logger
        try:
            if self.data_logger:
                self.data_logger.close()
        except Exception as e:
            self.logger.error(f"Error closing data logger: {e}")

        # Close logging handlers
        for handler in self.logger.handlers[:]:
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass

    async def _async_cleanup(self):
        """Async cleanup for aiohttp sessions and other async resources."""
        if self._cleanup_done:
            return

        self._cleanup_done = True

        # Close Maker client
        try:
            if self.maker_client:
                await asyncio.wait_for(
                    self.maker_client.disconnect(),
                    timeout=2.0
                )
                self.logger.info(f"🔌 {self.maker_exchange_name} client closed")
        except asyncio.TimeoutError:
            self.logger.warning(f"⚠️ Timeout closing {self.maker_exchange_name} client, forcing shutdown")
        except Exception as e:
            self.logger.error(f"Error closing Maker client: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def initialize_lighter_client(self):
        """Initialize the Lighter client."""
        if self.lighter_client is None:
            api_key_private_key = os.getenv('API_KEY_PRIVATE_KEY')
            if not api_key_private_key:
                raise Exception("API_KEY_PRIVATE_KEY environment variable not set")

            self.lighter_client = SignerClient(
                url=self.lighter_base_url,
                private_key=api_key_private_key,
                account_index=self.account_index,
                api_key_index=self.api_key_index,
            )

            err = self.lighter_client.check_client()
            if err is not None:
                raise Exception(f"CheckClient error: {err}")

            self.logger.info("✅ Lighter client initialized successfully")
        return self.lighter_client

    def initialize_maker_client(self):
        """Initialize the Maker client."""
        # Create a config dictionary for the factory
        config = {
            'ticker': self.ticker,
            'quantity': self.order_quantity,
            'tick_size': Decimal(0), # Will be updated
            'close_order_side': 'sell' # Placeholder
        }

        # region agent log
        import json, time as _t
        with open("/Users/junoshi/Desktop/cross-exchange-arbitrage-main/.cursor/debug.log", "a") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H1",
                "location": "maker_taker_arb.py:initialize_maker_client:before_factory",
                "message": "Creating maker client via factory",
                "data": {
                    "config_type": str(type(config)),
                    "config_keys": list(config.keys()),
                    "maker_exchange": self.maker_exchange_name
                },
                "timestamp": int(_t.time() * 1000)
            }) + "\n")
        # endregion

        self.maker_client = ExchangeFactory.create_exchange(self.maker_exchange_name, config)
        self.logger.info(f"✅ {self.maker_exchange_name} client initialized successfully")

        # region agent log
        with open("/Users/junoshi/Desktop/cross-exchange-arbitrage-main/.cursor/debug.log", "a") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H1",
                "location": "maker_taker_arb.py:initialize_maker_client:after_factory",
                "message": "Maker client created",
                "data": {
                    "maker_exchange": self.maker_exchange_name,
                    "client_class": str(type(self.maker_client))
                },
                "timestamp": int(_t.time() * 1000)
            }) + "\n")
        # endregion

        # Setup generic order update handler
        self.maker_client.setup_order_update_handler(self._handle_maker_order_update)
        
        return self.maker_client

    def get_lighter_market_config(self) -> Tuple[int, int, int, Decimal]:
        """Get Lighter market configuration."""
        url = f"{self.lighter_base_url}/api/v1/orderBooks"
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            if not response.text.strip():
                raise Exception("Empty response from Lighter API")

            data = response.json()

            if "order_books" not in data:
                raise Exception("Unexpected response format")

            for market in data["order_books"]:
                if market["symbol"] == self.ticker:
                    price_multiplier = pow(10, market["supported_price_decimals"])
                    return (market["market_id"],
                            pow(10, market["supported_size_decimals"]),
                            price_multiplier,
                            Decimal("1") / (Decimal("10") ** market["supported_price_decimals"]))
            raise Exception(f"Ticker {self.ticker} not found")

        except Exception as e:
            self.logger.error(f"⚠️ Error getting market config: {e}")
            raise

    async def trading_loop(self):
        """Main trading loop implementing the strategy."""
        self.logger.info(f"🚀 Starting arbitrage bot for {self.ticker} on {self.maker_exchange_name}")

        # Initialize clients
        try:
            self.initialize_lighter_client()
            self.initialize_maker_client()

            # Connect Maker Client (Handles WebSocket internally usually)
            # region agent log
            import json, time as _t
            with open("/Users/junoshi/Desktop/cross-exchange-arbitrage-main/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H3",
                    "location": "maker_taker_arb.py:trading_loop:before_maker_connect",
                    "message": "Connecting maker client",
                    "data": {"maker_exchange": self.maker_exchange_name},
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
            # endregion

            await self.maker_client.connect()

            # Get contract info
            self.maker_contract_id, self.maker_tick_size = await self.maker_client.get_contract_attributes()
            
            (self.lighter_market_index, self.base_amount_multiplier,
             self.price_multiplier, self.tick_size) = self.get_lighter_market_config()

            self.logger.info(
                f"Contract info loaded - {self.maker_exchange_name}: {self.maker_contract_id}, "
                f"Lighter: {self.lighter_market_index}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            self.logger.error(traceback.format_exc())
            return

        # Initialize position tracker
        # Note: Position tracker still expects edgex_client interface, but base exchange client should suffice 
        # if get_account_positions is implemented.
        # However, for simplicity, we pass the maker client and update PositionTracker if needed or trust duck typing
        # The current PositionTracker tracks 'edgex' specifically. We might need to update it or alias it.
        # For now, we manually update positions via 'update_edgex_position' method in tracker which is generic enough in logic.
        self.position_tracker = PositionTracker(
            self.ticker,
            self.maker_client, # Passing maker client
            self.maker_contract_id,
            self.lighter_base_url,
            self.account_index,
            self.logger
        )

        # Configure modules
        self.order_manager.set_maker_config(
            self.maker_client, self.maker_contract_id, self.maker_tick_size)
        self.order_manager.set_lighter_config(
            self.lighter_client, self.lighter_market_index,
            self.base_amount_multiplier, self.price_multiplier, self.tick_size)

        self.ws_manager.set_lighter_config(
            self.lighter_client, self.lighter_market_index, self.account_index)

        # Setup Lighter websocket
        try:
            self.ws_manager.start_lighter_websocket()
            self.logger.info("✅ Lighter WebSocket task started")

            # Wait for initial Lighter order book data
            self.logger.info("⏳ Waiting for initial Lighter order book data...")
            timeout = 10
            start_time = time.time()
            while (not self.order_book_manager.lighter_order_book_ready and
                   not self.stop_flag):
                if time.time() - start_time > timeout:
                    self.logger.warning(
                        f"⚠️ Timeout waiting for Lighter WebSocket order book data after {timeout}s")
                    break
                await asyncio.sleep(0.5)

            if self.order_book_manager.lighter_order_book_ready:
                self.logger.info("✅ Lighter WebSocket order book data received")
            else:
                self.logger.warning("⚠️ Lighter WebSocket order book not ready")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup Lighter websocket: {e}")
            return

        await asyncio.sleep(5)

        # Get initial positions
        # Note: PositionTracker.get_edgex_position() calls get_account_positions().
        # BaseExchangeClient requires get_account_positions().
        self.position_tracker.edgex_position = await self.position_tracker.get_edgex_position()
        self.position_tracker.lighter_position = await self.position_tracker.get_lighter_position()

        # Main trading loop
        while not self.stop_flag:
            try:
                # Fetch Maker BBO
                ex_best_bid, ex_best_ask = await asyncio.wait_for(
                    self.order_manager.fetch_maker_bbo_prices(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⚠️ Timeout fetching {self.maker_exchange_name} BBO prices")
                await asyncio.sleep(0.5)
                continue
            except Exception as e:
                self.logger.error(f"⚠️ Error fetching {self.maker_exchange_name} BBO prices: {e}")
                await asyncio.sleep(0.5)
                continue

            lighter_bid, lighter_ask = self.order_book_manager.get_lighter_bbo()

            # Determine if we should trade
            long_ex = False
            short_ex = False
            if (lighter_bid and ex_best_bid and
                    lighter_bid - ex_best_bid > self.long_ex_threshold):
                long_ex = True
            elif (ex_best_ask and lighter_ask and
                  ex_best_ask - lighter_ask > self.short_ex_threshold):
                short_ex = True

            # Log BBO data
            self.data_logger.log_bbo_to_csv(
                maker_bid=ex_best_bid,
                maker_ask=ex_best_ask,
                lighter_bid=lighter_bid if lighter_bid else Decimal('0'),
                lighter_ask=lighter_ask if lighter_ask else Decimal('0'),
                long_maker=long_ex,
                short_maker=short_ex,
                long_maker_threshold=self.long_ex_threshold,
                short_maker_threshold=self.short_ex_threshold
            )

            if self.stop_flag:
                break

            # Execute trades
            # Note: position_tracker.get_current_edgex_position returns the *cached* maker position
            if (self.position_tracker.get_current_edgex_position() < self.max_position and
                    long_ex):
                await self._execute_long_trade()
            elif (self.position_tracker.get_current_edgex_position() > -1 * self.max_position and
                  short_ex):
                await self._execute_short_trade()
            else:
                await asyncio.sleep(0.05)

    async def _execute_long_trade(self):
        """Execute a long trade (buy on Maker, sell on Lighter)."""
        if self.stop_flag:
            return

        # Update positions
        try:
            self.position_tracker.edgex_position = await asyncio.wait_for(
                self.position_tracker.get_edgex_position(),
                timeout=3.0
            )
            if self.stop_flag:
                return
            self.position_tracker.lighter_position = await asyncio.wait_for(
                self.position_tracker.get_lighter_position(),
                timeout=3.0
            )
        except asyncio.TimeoutError:
            if self.stop_flag:
                return
            self.logger.warning("⚠️ Timeout getting positions")
            return
        except Exception as e:
            if self.stop_flag:
                return
            self.logger.error(f"⚠️ Error getting positions: {e}")
            return

        if self.stop_flag:
            return

        self.logger.info(
            f"Maker position: {self.position_tracker.edgex_position} | "
            f"Lighter position: {self.position_tracker.lighter_position}")

        if abs(self.position_tracker.get_net_position()) > self.order_quantity * 2:
            self.logger.error(
                f"❌ Position diff is too large: {self.position_tracker.get_net_position()}")
            sys.exit(1)

        self.order_manager.order_execution_complete = False
        self.order_manager.waiting_for_lighter_fill = False

        try:
            side = 'buy'
            order_filled = await self.order_manager.place_maker_post_only_order(
                side, self.order_quantity, self.stop_flag)
            if not order_filled or self.stop_flag:
                return
        except Exception as e:
            if self.stop_flag:
                return
            self.logger.error(f"⚠️ Error in trading loop: {e}")
            self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
            sys.exit(1)

        start_time = time.time()
        while not self.order_manager.order_execution_complete and not self.stop_flag:
            if self.order_manager.waiting_for_lighter_fill:
                await self.order_manager.place_lighter_market_order(
                    self.order_manager.current_lighter_side,
                    self.order_manager.current_lighter_quantity,
                    self.order_manager.current_lighter_price,
                    self.stop_flag
                )
                break

            await asyncio.sleep(0.01)
            if time.time() - start_time > 180:
                self.logger.error("❌ Timeout waiting for trade completion")
                break

    async def _execute_short_trade(self):
        """Execute a short trade (sell on Maker, buy on Lighter)."""
        if self.stop_flag:
            return

        # Update positions
        try:
            self.position_tracker.edgex_position = await asyncio.wait_for(
                self.position_tracker.get_edgex_position(),
                timeout=3.0
            )
            if self.stop_flag:
                return
            self.position_tracker.lighter_position = await asyncio.wait_for(
                self.position_tracker.get_lighter_position(),
                timeout=3.0
            )
        except asyncio.TimeoutError:
            if self.stop_flag:
                return
            self.logger.warning("⚠️ Timeout getting positions")
            return
        except Exception as e:
            if self.stop_flag:
                return
            self.logger.error(f"⚠️ Error getting positions: {e}")
            return

        if self.stop_flag:
            return

        self.logger.info(
            f"Maker position: {self.position_tracker.edgex_position} | "
            f"Lighter position: {self.position_tracker.lighter_position}")

        if abs(self.position_tracker.get_net_position()) > self.order_quantity * 2:
            self.logger.error(
                f"❌ Position diff is too large: {self.position_tracker.get_net_position()}")
            sys.exit(1)

        self.order_manager.order_execution_complete = False
        self.order_manager.waiting_for_lighter_fill = False

        try:
            side = 'sell'
            order_filled = await self.order_manager.place_maker_post_only_order(
                side, self.order_quantity, self.stop_flag)
            if not order_filled or self.stop_flag:
                return
        except Exception as e:
            if self.stop_flag:
                return
            self.logger.error(f"⚠️ Error in trading loop: {e}")
            self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
            sys.exit(1)

        start_time = time.time()
        while not self.order_manager.order_execution_complete and not self.stop_flag:
            if self.order_manager.waiting_for_lighter_fill:
                await self.order_manager.place_lighter_market_order(
                    self.order_manager.current_lighter_side,
                    self.order_manager.current_lighter_quantity,
                    self.order_manager.current_lighter_price,
                    self.stop_flag
                )
                break

            await asyncio.sleep(0.01)
            if time.time() - start_time > 180:
                self.logger.error("❌ Timeout waiting for trade completion")
                break

    async def run(self):
        """Run the arbitrage bot."""
        self.setup_signal_handlers()

        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Received interrupt signal...")
        except asyncio.CancelledError:
            self.logger.info("\n🛑 Task cancelled...")
        finally:
            self.logger.info("🔄 Cleaning up...")
            self.shutdown()
            # Ensure async cleanup is done with timeout
            try:
                await asyncio.wait_for(self._async_cleanup(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Cleanup timeout, forcing exit")
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
