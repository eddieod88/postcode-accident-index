import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from typing import Dict, List, Optional, Set


class PostcodeField(Enum):
    POSTCODE = "postcode"
    LONG = "longitude"
    LAT = "latitude"
    ITL_CODE = "codes.nuts"


@dataclass
class PostcodePayload:
    fields: List[PostcodeField]

    def __post_init__(self):
        if PostcodeField.POSTCODE not in self.fields:
            self.fields.append(PostcodeField.POSTCODE)
        self.nested_value = "codes"

    def get_payload_params(self) -> Dict[str, str]:
        filter_string = ""
        for f in self.fields:
            if self.nested_value in f.value:
                filter_string += self.nested_value + ","
            else:
                filter_string += f.value + ","
        if filter_string[-1] == ",":
            filter_string = filter_string[:-1]

        return {"filter": filter_string}

    def extract_response(self, response: Dict) -> Dict:
        results = []
        for r in response["result"]:
            result = {}
            for f in self.fields:
                if self.nested_value in f.value:
                    keys = f.value.split(".")
                    result[f.value] = r["result"][keys[0]][keys[1]]
                else:
                    result[f.value] = r["result"][f.value]
            results.append(result)
        return results


class PostcodeClient:
    endpoint = "https://api.postcodes.io"

    async def fetch_locations(self, postcodes: Set[str], fields: PostcodePayload) -> List[Dict]:
        postcodes = list(postcodes)
        batch_size = 100  # API limitation

        batches = asyncio.gather(
            *[
                self._get(postcodes[i : i + batch_size], fields)
                for i in range(0, len(postcodes), batch_size)
            ],
            return_exceptions=False,
        )
        # expand batches into individual results
        results = [result for batch in await batches for result in batch]

        return results

    async def _get(
        self,
        postcodes: Set[str],
        payload: PostcodePayload,
    ) -> List[Dict]:
        """
        Get locations for up to 100 postcodes
        """
        assert (
            batch_length := len(postcodes)
        ) <= 100, "API cannot accept more than 100 postcodes at a time"

        filters = payload.get_payload_params()
        endpoint = self.endpoint + "/postcodes"
        lookup_data = {"postcodes": postcodes}
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, data=lookup_data, params=filters) as resp:
                logger.info(f"Sending request to postcodes.io ({batch_length})")
                response = await resp.json()
        if response["status"] == 200:
            locations = payload.extract_response(response)
        else:
            response.raise_for_status()

        return locations
