import sys
from poa.pipline import *
from poa.es import *
import copy
import jieba
import re
import os


class OnlineTestCorpus(object):

    def __init__(self, conf):
        self.company_list = conf['company_list']
        self.fullname2shortname = self.fullname2shortname()
        self.ambiguous_fullname = [c['full_name'] for c in self.company_list]
        self.ambiguous_shortname = [c['short_name'] for c in self.company_list]
        self.path = conf['online_pos']
        self.date = conf['online_date']

        print(self.fullname2shortname)

    def fullname2shortname(self):
        dic = {}
        for c in self.company_list:
            dic[c['full_name']] = c['short_name']
        return dic

    def read_es(self, start_date, end_date):
        query = {
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "relation": "主体"
                                    }
                                },
                                {
                                    "term": {
                                        "source": "news"
                                    }
                                },
                                {
                                    "range": {
                                        "publish_time": {
                                            "gte": "2018-08-20 00:00:00",
                                            "lte": "2018-08-21 00:00:00"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
        query['query']['bool']['filter']['bool']['must'][2]['range']['publish_time']['gte'] = start_date+' 00:00:00'
        query['query']['bool']['filter']['bool']['must'][2]['range']['publish_time']['lte'] = end_date + ' 00:00:00'
        esindex = es_index('192.168.2.46', 9200, 'online2_news_index', 'online_news')
        docs = esindex.search_index_scroll(query)
        print("total doc cnt:%d" % (len(docs)))
        docs_ambiguous = [doc for doc in docs if doc['company_name'] in self.ambiguous_fullname]
        print("ambiguous docs cnt: %d"%len(docs_ambiguous))
        return docs_ambiguous

    def collect_online_pos(self):
        docs = self.read_es(self.date['start'], self.date['end'])
        with open(self.path+'online_pos/'+self.date['start']+'.pos', 'w+') as f:
            for doc in docs:
                short_name = self.fullname2shortname[doc['company_name']]
                title = doc['title']
                title = re.sub('\t|\n', ' ', title)
                label = '1'
                f.write(short_name+'\t'+title+'\t'+label+'\n')
