import os

import preset

fpath_index_json = os.path.join(preset.dpath_data_raw, r'ADT_download_urls.json')


dpath_cache_aria_adt = os.path.join(preset.dpath_data_cache, r'aria_adt')
os.makedirs(dpath_cache_aria_adt, exist_ok=True)
