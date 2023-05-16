import redis

# 连接 Redis 服务器
r = redis.Redis(host='localhost', port=6379)

# 在分布式哈希表中设置键值对
r.hset('myhash', 'key1', 'value1')
r.hset('myhash', 'key2', 'value2')
r.hset('myhash', 'key3', 'value3')

# 获取分布式哈希表中的值
print("r:",r)
print(r.hget('myhash', 'key1'))
print(r.hget('myhash', 'key2'))
print(r.hget('myhash', 'key3'))
