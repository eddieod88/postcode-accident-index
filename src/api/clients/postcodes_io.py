import asyncio
import aiohttp
from enum import Enum
from loguru import logger
from typing import Dict, List, Optional, Set


class PostcodeField(Enum):
    POSTCODE = "postcode"
    LONG = "longitude"
    LAT = "latitude"


class PostcodeClient:
    endpoint = "https://api.postcodes.io"

    async def fetch_locations(self, postcodes: Set[str]) -> List[Dict]:
        postcodes = list(postcodes)
        batch_size = 100  # API limitation
        fields = [PostcodeField.LONG, PostcodeField.LAT]

        batches = asyncio.gather(
            *[
                self._get(postcodes[i : i + batch_size], fields=fields)
                for i in range(0, len(postcodes), batch_size)
            ],
            return_exceptions=False,
        )
        # expand batches into individual results
        results = [result for batch in await batches for result in batch]

        return results

    async def _get(
        self, postcodes: Set[str], fields: Optional[List[PostcodeField]] = None
    ) -> List[Dict]:
        """
        Get locations for up to 100 postcodes
        """
        assert (
            batch_length := len(postcodes)
        ) <= 100, "API cannot accept more than 100 postcodes at a time"

        filters = None
        if fields:
            if PostcodeField.POSTCODE not in fields:
                fields.append(PostcodeField.POSTCODE)
            filters = {"filter": ",".join([f.value for f in fields])}

        endpoint = self.endpoint + "/postcodes"
        payload = {"postcodes": postcodes}
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, data=payload, params=filters) as resp:
                logger.info(f"Sending request to postcodes.io ({batch_length})")
                response = await resp.json()
        if response["status"] == 200:
            locations = [
                {f.value: r["result"][f.value] for f in fields} for r in response["result"]
            ]
        else:
            response.raise_for_status()

        return locations
