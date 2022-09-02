# coding=utf-8
import requests
import time

params = {
    "text": "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"
}

url = "http://0.0.0.0:9277/extraction/"
start = time.time()
res = requests.post(url=url, json=params)
print(res.text)
end = time.time()
print("耗时：{}s".format(end - start))