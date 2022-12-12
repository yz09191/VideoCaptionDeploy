import requests

res = requests.delete('http://114.212.87.252:3001/models/navc/')
print(res.content)
requests.post('http://114.212.87.252:3001/models?url=navc.mar')
print(res.content)
requests.put('http://114.212.87.252:3001/models/navc?min_worker=1&synchronous=true')
print(res.content)