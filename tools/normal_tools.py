from langchain.tools import tool

@tool
def get_weather(location:str) -> str:
    """
    用于获取某个地方的天气

    Args:
        location(str): 地点
    """
    return f"{location} 气温25°，整日晴朗"
