import logging


def setup_custom_logger(name):
    """
    公用日志配置
    :param name: 日志某块名
    :return: logger
    """
    # 设置日志格式
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)8s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    # 设置日志等级
    logger.setLevel(logging.DEBUG)
    return logger
