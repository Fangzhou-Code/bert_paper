'''
作用：存储从区块链传递过来的数据，同时采用范围分片技术进行检索和查询
'''
print("=====Distributed Hash Table======")
import redis

def storage(data,host,port):
    print("=====storage redis======")
    # 连接 Redis 服务器
    #r = redis.Redis(host='localhost', port=6379)
    r = redis.Redis(host, port)
    # 在分布式哈希表中设置键值对
    '''
    HSET key field value 
    其中，key表示 hash 对象的名称，field表示要设置的字段名，value表示要设置的值。如果 key 不存在，则创建一个新的 hash 对象。
    如果 field 已经存在，那么原来的值会被覆盖。如果 value 的类型不是字符串，那么它将被转换为字符串后再存储。
    该命令返回值为：
        如果 field 是一个新字段，并且 value 已经成功设置，那么返回 1。
        如果 field 已经存在，并且 value 已经成功更新，那么返回 0。
    '''
    print("hset:", r.hset('myhash', 'key1', data))
    print("=====end=====")

def rangeSharding(host, port):
    print("=====rangeSharding======")
    # 获取分布式哈希表中的值
    r = redis.Redis(host, port)
    print("r:",r)
    print("hget:",r.hget('myhash', 'key1'))
    print("=====end=====")

print("=====end=====")