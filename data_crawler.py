import urllib.request as rq
import time
from bs4 import BeautifulSoup


def get_url_content(url, count=0):
    # 构造发送请求
    if count > 2:
        return "111"
    try:
        request = rq.Request(url)

        # 发出请求并取得响应
        response = rq.urlopen(request)

        # 获取网页内容
        html = response.read()

        # 返回网页内容
        return html
    except:
        time.sleep(60)
        get_url_content(url, count + 1)


def getdr(tit):
    constr = ""
    for cha in tit:
        if cha.isdigit():
            constr += cha
    if len(constr) < 6:
        constr = constr[:4] + "0" + constr[-1]
    return constr


def getCount(tit):
    constr = ""
    for cha in tit:
        if cha.isdigit():
            constr += cha
    if len(constr) > 8:
        return ""
    return constr


from collections import defaultdict
import pandas as pd
import re

if __name__ == "__main__":
    xs_a = ['增量指标', "参与竞价的有效编码", "有效编码", "成交编码", "两次平均报价", "最低成交价的报价人数", "最低成交价", "平均成交价"]
    p = ['个人和单位', '个人', '单位']
    drinfo = defaultdict(lambda: defaultdict(lambda: list()))
    for i in range(1, 14):
        if i == 1:
            url = "http://xqctk.sztb.gov.cn/gbl/index.html"
        else:
            url = "http://xqctk.sztb.gov.cn/gbl/index_%d.html" % i
        content_1 = BeautifulSoup(get_url_content(url), "html.parser")
        for one in content_1.find(class_="blist").find_all('dd'):
            if "小汽车增量指标竞价情况" in one.find("a").text:
                title = one.find("a").text
                dr = getdr(title)
                print(dr)
                href = one.find("a").get("href")
                content_2 = BeautifulSoup(get_url_content(href), "html.parser")
                infoarr = content_2.find(class_='details').text
                ss = infoarr.strip().replace("\t", "").split("。")
                for info_one in ss:
                    ss_dh = re.split(r'[：、；，]', info_one)
                    print(ss_dh)
                    for i in range(len(ss_dh)):
                        if "时间" in ss_dh[i] or "小时" in ss_dh[i] or "." in ss_dh[i]:
                            continue
                        digital_str = getCount(ss_dh[i])
                        mc = ""
                        xs = ""
                        if len(digital_str) > 0:
                            print(digital_str)
                            if digital_str == "56713":
                                print(digital_str)
                            if i == 0:
                                for xs_one in xs_a:
                                    if xs_one in ss_dh[i]:
                                        xs = xs_one
                                        break
                                for mc_one in p:
                                    if mc_one in ss_dh[i]:
                                        mc = mc_one
                                        break
                            else:
                                for j in range(0, i + 1):
                                    if len(mc) > 0 and len(xs) > 0:
                                        break
                                    if mc == "":
                                        for mc_one in p:
                                            if mc_one in ss_dh[i - j]:
                                                mc = mc_one
                                                break
                                    if xs == "":
                                        for xs_one in xs_a:
                                            if xs_one in ss_dh[i - j]:
                                                xs = xs_one
                                                break
                        if len(mc) > 0 and len(xs) > 0:
                            drinfo[dr][mc + xs].append(digital_str)

    data = pd.DataFrame(drinfo).T
    print(drinfo['201808'])
    data.to_excel("/Users/a/Documents/sz_jp_info.xlsx")
