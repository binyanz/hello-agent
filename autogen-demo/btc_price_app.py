import streamlit as st
import requests
from datetime import datetime

# Streamlit 应用标题
st.title("比特币价格显示应用")

# 设置 API 地址
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"

# 获取比特币价格数据
def get_btc_price():
    try:
        response = requests.get(f"{COINGECKO_API_URL}?ids=bitcoin&vs_currencies=usd")
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        price = data['bitcoin']['usd']
        return price
    except requests.RequestException as e:
        st.error(f"请求失败: {e}")
        return None

# 获取24小时涨跌幅和涨跌额
def get_btc_trend():
    try:
        response = requests.get(f"{COINGECKO_API_URL}?ids=bitcoin&vs_currencies=usd&include_24hr_change=true")
        response.raise_for_status()
        data = response.json()
        price = data['bitcoin']['usd']
        change_24h = data['bitcoin']['usd_24h_change']
        return price, change_24h
    except requests.RequestException as e:
        st.error(f"请求失败: {e}")
        return None, None

# 显示比特币价格和24小时价格变化趋势
def display_btc_price(price, change_24h):
    st.header("比特币当前价格")
    st.write(f"当前价格: ${price}")
    
    st.header("24小时价格变化趋势")
    if change_24h > 0:
        st.success(f"涨 ${abs(change_24h)} (涨幅 {change_24h:.2f}%)")
    elif change_24h < 0:
        st.error(f"跌 ${abs(change_24h)} (跌幅 {abs(change_24h):.2f}%)")
    else:
        st.info("价格保持不变")
        
# 刷新按钮
if st.button("刷新"):
    price, change_24h = get_btc_trend()
    if price and change_24h:
        display_btc_price(price, change_24h)
    else:
        st.error("无法获取最新数据，请稍后再试")

# 显示加载状态
if st.button("显示加载状态"):
    st.write(f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 运行应用
if __name__ == "__main__":
    st.write("欢迎使用比特币价格显示应用！请使用界面按钮进行操作。")