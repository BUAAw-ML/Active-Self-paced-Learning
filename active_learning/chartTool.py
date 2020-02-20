# -*- coding: UTF-8 -*-
import json
import urllib3, urllib


global group_name
group_name = "[5.25-?]CNN+PFS-18000+MRR+5000+150"

'''
No-Deterministic
Deterministic
Random
'''
global sample_method
sample_method = "Random+seed-0"

global max_performance
max_performance = "0"

global bar_legend
bar_legend = 0


# Update line chart
def updateLineChart(mrr, sm, gp_name = group_name, max = max_performance):

    http = urllib3.PoolManager()

    global sample_method
    global max_performance

    if max == 0:
        max = max_performance

    info = {}
    info['active_combine'] = gp_name
    info['sample_method'] = sm
    info['max_performance'] = max
    info['bar_legend'] = bar_legend
    info["mrr_data"] = mrr
    info["type"] = 1

    refs = urllib.parse.urlencode(info)

    urls = '10.2.26.117/updateChart/?' + refs
    http.request('GET', urls)

# Update histogram
def updateBarChart(data):

    http = urllib3.PoolManager()

    global active_combine
    global sample_method

    info = {}
    info['active_combine'] = active_combine
    info['sample_method'] = sample_method
    info['max_performance'] = max_performance
    info['bar_legend'] = bar_legend
    info["data"] = data
    info["type"] = 2

    refs = urllib.parse.urlencode(info)

    urls = '10.2.26.117/updateChart/?' + refs
    http.request('GET', urls)