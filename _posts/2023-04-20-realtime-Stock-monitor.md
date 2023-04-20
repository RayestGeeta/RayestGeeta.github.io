---
title: 股市&汇率实时监控(stock monitor)
author: Rayest
date: 2023-04-20 17:43:00 +0800
categories: [Tools]
tags: [technology]
math: true
---

## 后续更新：基础代码已完成后

```python
import re
import requests
from bs4 import BeautifulSoup
import efinance as ef
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import schedule
import time

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

init_money = 80000
init_buy = 134
nums = 679
stock_code = '03690'
freq = 30

initial_buy = 689.52/100
dollar_num = 5000

def schedule_run():
    # 监控美元、港元汇率
    url = 'http://www.boc.cn/sourcedb/whpj/index.html'
    html = requests.get(url).content.decode('utf8')
    soup=BeautifulSoup(html,'lxml')


    dollar_index = html.index('<td>美元</td>')
    dollar_html = html[dollar_index:dollar_index + 500]
    dollar_info = re.findall('<td>(.*?)</td>', dollar_html)
    push_date = re.findall('<td class="pjrq">(.*?)</td>', dollar_html)
    result_sell = dollar_info[1]

    hk_index = html.index('<td>港币</td>')
    hk_html = html[hk_index:hk_index + 500]
    hk_info = re.findall('<td>(.*?)</td>', hk_html)
    # push_date = re.findall('<td class="pjrq">(.*?)</td>', dollar_html)
    hk_result_sell = hk_info[1]


    initial_rmb = initial_buy * dollar_num
    result_rmb = float(result_sell)/100 * dollar_num
    income = round((result_rmb-initial_rmb)*100/initial_rmb, 2)


    # 获取实时美团股价
    df = ef.stock.get_quote_history(stock_code, klt=freq)

    def process(x):
        return ''.join(x.split()[0].split('-')[1:]) + '-' + x.split()[1].split(':')[0]

    df.index = df['日期'].apply(process)
    df['收盘(rmb)'] = float(hk_result_sell)/100*df['收盘']
    df['股票金额(rmb)'] =  df['收盘(rmb)']*nums
    df['收益(百分点)'] = round(100*(df['股票金额(rmb)']-init_money)/init_money,4)


    # df[['收益(百分点)']].plot(secondary_y=['收益(百分点)'])
    # df[['收益(百分点)']].plot()

    save_data = list(df.iloc[-1][['日期', '股票名称','股票代码','开盘','收盘', '收盘(rmb)', '股票金额(rmb)', '收益(百分点)']].values)
    columns = ['日期', '股票名称','股票代码','开盘(hk)','收盘(hk)', '收盘(rmb)', '股票金额(rmb)', '股票收益(百分点)','初始股票(rmb)', '股票收益(rmb)', 
               '初始购买现汇', '现在卖出现汇', '汇率收益(百分比)', '汇率收益(rmb)', '初始股票价格(rmb)', '初始美元汇率', '现在美元汇率', '初始港币汇率', '现在港币汇率']
    save_data.append(80000)
    save_data.append(round(save_data[-3]-80000,2))
    save_data.append(initial_rmb)
    save_data.append(result_rmb)
    save_data.append(income)
    save_data.append(result_rmb-initial_rmb)
    save_data.append(117)
    save_data.append(689.52)
    save_data.append(result_sell)
    save_data.append(87.80)
    save_data.append(hk_result_sell)

    save_df= pd.read_csv('实时监控.csv')
    # first save
    # save_df = pd.DataFrame(np.array([save_data]), columns=columns)
    # save_df.to_csv('实时监控.csv', index=False)

    save_df= pd.read_csv('实时监控.csv')
    save_new_df = pd.DataFrame(np.array([save_data]), columns=columns)
    save_df.append(save_new_df)
    save_df.to_csv('实时监控.csv', index=False)

    with open('log.txt','a') as file0:
        print('当前时间:', push_date[0],file=file0)
        print('初始购买现汇(' + str(689.52) +')花费rmb:', initial_rmb,file=file0)
        print('现在卖出现汇(' + str(result_sell) +')到手rmb:', result_rmb,file=file0)
        print('汇率收益:', str(income) + '%(' + str(result_rmb-initial_rmb) + ')',file=file0)
        print('',file=file0)

        print('初始股票(' + str(117.65) +')金额:', 80000,file=file0)
        print('现在股票(' + str(round(df.iloc[-1][-3],2)) +')金额:', round(df.iloc[-1][-2],0),file=file0)
        print('股票收益:', str(df.iloc[-1][-1]) + '%(' + str(round(df.iloc[-1][-2],0) - 80000) + ')',file=file0)
        print('\n\n', file=file0)
        
def time_now():
    print('股市监控 时间点:' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' 结束.')

schedule.every(30).minutes.do(schedule_run)
schedule.every(30).minutes.do(time_now)

# schedule.every(10).seconds.do(schedule_run)
while True:
    schedule.run_pending() # 运行所有可运行的任务
    time.sleep(1)
```
