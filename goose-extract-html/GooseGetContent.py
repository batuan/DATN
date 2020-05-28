from goose3 import Goose
from goose3.configuration import Configuration
import json
import re
config = Configuration()
config.strict = False  # turn of strict exception handling
config.browser_user_agent = 'Mozilla 5.0'  # set the browser agent string
config.http_timeout = 5.05  # set http timeout in seconds
with Goose(config) as g:
    pass

def getContent(url):
    rs = None
    try:
        content = g.extract(url)
        rs = {
            "url": url,
            "type": None,
            "title": content.title,
            "description": content.meta_description
        }
    except:
        print("Can't get content error url")
    return rs