"""Position tracking for Maker (Generic) and Lighter exchanges."""
import asyncio
import json
import logging
import requests
import sys
from decimal import Decimal


class PositionTracker:
    """Tracks positions on both exchanges."""

    def __init__(self, ticker: str, maker_client, maker_contract_id: str,
                 lighter_base_url: str, account_index: int, logger: logging.Logger):
        """Initialize position tracker."""
        self.ticker = ticker
        self.maker_client = maker_client
        self.maker_contract_id = maker_contract_id
        self.lighter_base_url = lighter_base_url
        self.account_index = account_index
        self.logger = logger

        self.maker_position = Decimal('0')
        self.lighter_position = Decimal('0')

    async def get_maker_position(self) -> Decimal:
        """Get Maker position."""
        if not self.maker_client:
            raise Exception("Maker client not initialized")

        try:
            # BaseExchangeClient guarantees returning a Decimal for get_account_positions
            position = await self.maker_client.get_account_positions()
            return position
        except Exception as e:
            self.logger.warning(f"Failed to get Maker positions: {e}")
            return Decimal('0')

    async def get_lighter_position(self) -> Decimal:
        """Get Lighter position."""
        url = f"{self.lighter_base_url}/api/v1/account"
        headers = {"accept": "application/json"}

        current_position = None
        parameters = {"by": "index", "value": self.account_index}
        attempts = 0
        while current_position is None and attempts < 10:
            try:
                response = requests.get(url, headers=headers, params=parameters, timeout=10)
                response.raise_for_status()

                if not response.text.strip():
                    self.logger.warning("⚠️ Empty response from Lighter API for position check")
                    return self.lighter_position

                data = response.json()

                if 'accounts' not in data or not data['accounts']:
                    self.logger.warning(f"⚠️ Unexpected response format from Lighter API: {data}")
                    return self.lighter_position

                positions = data['accounts'][0].get('positions', [])
                for position in positions:
                    if position.get('symbol') == self.ticker:
                        current_position = Decimal(position['position']) * position['sign']
                        break
                if current_position is None:
                    current_position = Decimal('0')

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"⚠️ Network error getting position: {e}")
            except json.JSONDecodeError as e:
                self.logger.warning(f"⚠️ JSON parsing error in position response: {e}")
                self.logger.warning(f"Response text: {response.text[:200]}...")
            except Exception as e:
                self.logger.warning(f"⚠️ Unexpected error getting position: {e}")
            finally:
                attempts += 1
                if current_position is None:
                    await asyncio.sleep(1)

        if current_position is None:
            self.logger.error(f"❌ Failed to get Lighter position after {attempts} attempts")
            sys.exit(1)

        return current_position

    def update_maker_position(self, delta: Decimal):
        """Update Maker position by delta."""
        self.maker_position += delta

    def update_lighter_position(self, delta: Decimal):
        """Update Lighter position by delta."""
        self.lighter_position += delta

    def get_current_maker_position(self) -> Decimal:
        """Get current Maker position (cached)."""
        return self.maker_position

    def get_current_lighter_position(self) -> Decimal:
        """Get current Lighter position (cached)."""
        return self.lighter_position

    def get_net_position(self) -> Decimal:
        """Get net position across both exchanges."""
        return self.maker_position + self.lighter_position
