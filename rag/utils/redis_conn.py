import logging
import json
import time
import uuid

import valkey as redis
from rag import settings
from rag.utils import singleton


class Payload:
    def __init__(self, consumer, queue_name, group_name, msg_id, message):
        self.__consumer = consumer
        self.__queue_name = queue_name
        self.__group_name = group_name
        self.__msg_id = msg_id
        self.__message = json.loads(message["message"])

    def ack(self):
        try:
            self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]ack" + str(self.__queue_name) + "||" + str(e))
        return False

    def get_message(self):
        return self.__message


@singleton
class RedisDB:
    def __init__(self):
        self.REDIS = None
        self.config = settings.REDIS
        self.__open__()

    def __open__(self):
        try:
            self.REDIS = redis.StrictRedis(
                host=self.config["host"].split(":")[0],
                port=int(self.config.get("host", ":6379").split(":")[1]),
                db=int(self.config.get("db", 1)),
                password=self.config.get("password"),
                decode_responses=True,
            )
        except Exception:
            logging.warning("Redis can't be connected.")
        return self.REDIS

    def health(self):
        self.REDIS.ping()
        a, b = "xx", "yy"
        self.REDIS.set(a, b, 3)

        if self.REDIS.get(a) == b:
            return True

    def is_alive(self):
        return self.REDIS is not None

    def exist(self, k):
        if not self.REDIS:
            return
        try:
            return self.REDIS.exists(k)
        except Exception as e:
            logging.warning("RedisDB.exist " + str(k) + " got exception: " + str(e))
            self.__open__()

    def get(self, k):
        if not self.REDIS:
            return
        try:
            return self.REDIS.get(k)
        except Exception as e:
            logging.warning("RedisDB.get " + str(k) + " got exception: " + str(e))
            self.__open__()

    def set_obj(self, k, obj, exp=3600):
        try:
            self.REDIS.set(k, json.dumps(obj, ensure_ascii=False), exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set_obj " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def set(self, k, v, exp=3600):
        try:
            self.REDIS.set(k, v, exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def sadd(self, key: str, member: str):
        try:
            self.REDIS.sadd(key, member)
            return True
        except Exception as e:
            logging.warning("RedisDB.sadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def srem(self, key: str, member: str):
        try:
            self.REDIS.srem(key, member)
            return True
        except Exception as e:
            logging.warning("RedisDB.srem " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def smembers(self, key: str):
        try:
            res = self.REDIS.smembers(key)
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.smembers " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def zadd(self, key: str, member: str, score: float):
        try:
            self.REDIS.zadd(key, {member: score})
            return True
        except Exception as e:
            logging.warning("RedisDB.zadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def zcount(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zcount(key, min, max)
            return res
        except Exception as e:
            logging.warning("RedisDB.zcount " + str(key) + " got exception: " + str(e))
            self.__open__()
        return 0

    def zpopmin(self, key: str, count: int):
        try:
            res = self.REDIS.zpopmin(key, count)
            return res
        except Exception as e:
            logging.warning("RedisDB.zpopmin " + str(key) + " got exception: " + str(e))
            self.__open__()
        return None

    def zrangebyscore(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zrangebyscore(key, min, max)
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.zrangebyscore " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def transaction(self, key, value, exp=3600):
        try:
            pipeline = self.REDIS.pipeline(transaction=True)
            pipeline.set(key, value, exp, nx=True)
            pipeline.execute()
            return True
        except Exception as e:
            logging.warning(
                "RedisDB.transaction " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return False

    def queue_product(self, queue, message, exp=settings.SVR_QUEUE_RETENTION) -> bool:
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                pipeline = self.REDIS.pipeline()
                pipeline.xadd(queue, payload)
                # pipeline.expire(queue, exp)
                pipeline.execute()
                return True
            except Exception as e:
                logging.exception(
                    "RedisDB.queue_product " + str(queue) + " got exception: " + str(e)
                )
        return False

    def queue_consumer(
        self, queue_name, group_name, consumer_name, msg_id=b">"
    ) -> Payload:
        try:
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
            args = {
                "groupname": group_name,
                "consumername": consumer_name,
                "count": 1,
                "block": 10000,
                "streams": {queue_name: msg_id},
            }
            messages = self.REDIS.xreadgroup(**args)
            if not messages:
                return None
            stream, element_list = messages[0]
            msg_id, payload = element_list[0]
            res = Payload(self.REDIS, queue_name, group_name, msg_id, payload)
            return res
        except Exception as e:
            if "key" in str(e):
                pass
            else:
                logging.exception(
                    "RedisDB.queue_consumer "
                    + str(queue_name)
                    + " got exception: "
                    + str(e)
                )
        return None

    def get_unacked_for(self, consumer_name, queue_name, group_name):
        try:
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                return
            pendings = self.REDIS.xpending_range(
                queue_name,
                group_name,
                min=0,
                max=10000000000000,
                count=1,
                consumername=consumer_name,
            )
            if not pendings:
                return
            msg_id = pendings[0]["message_id"]
            msg = self.REDIS.xrange(queue_name, min=msg_id, count=1)
            _, payload = msg[0]
            return Payload(self.REDIS, queue_name, group_name, msg_id, payload)
        except Exception as e:
            if "key" in str(e):
                return
            logging.exception(
                "RedisDB.get_unacked_for " + consumer_name + " got exception: " + str(e)
            )
            self.__open__()

    def queue_info(self, queue, group_name) -> dict | None:
        try:
            groups = self.REDIS.xinfo_groups(queue)
            for group in groups:
                if group["name"] == group_name:
                    return group
        except Exception as e:
            logging.warning(
                "RedisDB.queue_info " + str(queue) + " got exception: " + str(e)
            )
        return None


REDIS_CONN = RedisDB()


class RedisDistributedLock:
    def __init__(self, lock_key, ttl=60*40, timeout=10):
        self.lock_key = lock_key
        self.lock_value = str(uuid.uuid4())
        self.ttl = ttl
        self.acquire_timeout = timeout
        self._lock_acquired = False

    @staticmethod
    def clean_lock(lock_key):
        REDIS_CONN.REDIS.delete(lock_key)

    def acquire_lock(self):
        end_time = time.time() + self.acquire_timeout
        attempt = 0
        while time.time() < end_time:
            attempt += 1
            acquired = REDIS_CONN.REDIS.set(
                self.lock_key, 
                self.lock_value, 
                nx=True, 
                ex=self.ttl
            )
            if acquired:
                logging.debug(f"Lock acquired for {self.lock_key} (attempt {attempt})")
                self._lock_acquired = True
                return True
            
            current_value = REDIS_CONN.REDIS.get(self.lock_key)
            ttl_remaining = REDIS_CONN.REDIS.ttl(self.lock_key)
            logging.warning(
                f"Lock contention detected for {self.lock_key} | "
                f"Current holder: {current_value} | "
                f"TTL remaining: {ttl_remaining}s | "
                f"Attempt {attempt}, retrying in 0.1s"
            )
            
            time.sleep(max(0.1, (end_time - time.time()) / 10))
        
        current_value = REDIS_CONN.REDIS.get(self.lock_key)
        ttl_remaining = REDIS_CONN.REDIS.ttl(self.lock_key)
        clients = REDIS_CONN.REDIS.client_list()
        
        logging.error(
            f"Failed to acquire lock {self.lock_key} after {self.acquire_timeout}s | "
            f"Current holder: {current_value} | "
            f"TTL remaining: {ttl_remaining}s | "
            f"Connected clients: {len(clients)}"
        )
        return False

    def release_lock(self):
        if not self._lock_acquired:
            logging.warning(f"Attempted to release unacquired lock: {self.lock_key}")
            return
            
        logging.info(f"Releasing lock {self.lock_key}")
        lua_script = """
        if redis.call("get",KEYS[1]) == ARGV[1] then
            return redis.call("del",KEYS[1])
        else
            return 0
        end"""
        try:
            result = REDIS_CONN.REDIS.eval(lua_script, 1, self.lock_key, self.lock_value)
            if result == 1:
                logging.info(f"Lock {self.lock_key} released successfully")
            else:
                current_value = REDIS_CONN.REDIS.get(self.lock_key)
                logging.error(
                    f"Failed to release lock {self.lock_key} | "
                    f"Expected value: {self.lock_value} | "
                    f"Actual value: {current_value}"
                )
        except Exception as e:
            logging.error(f"Release lock failed: {str(e)}")
        finally:
            self._lock_acquired = False

    def __enter__(self):
        if self.acquire_lock():
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_lock()