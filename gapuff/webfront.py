#coding=utf-8
'''
Created on 2011-11-11
这个模块是模型的web接口，数据传输接口也是这里
@author: sugar
'''

import web
import multi_puff
from multi_puff import model_puff_core

urls = ("/", "hello",
        "/webrun", "run"
        )
app = web.application(urls, globals())

class hello:
    def GET(self):
        web.Redirect("/static/submit.htm", absolute=True)

class run:

    def POST(self):
        input = web.input()
        #Generate release sequence and met condition sequence
        for i in range(1,11):
            key11 = "time" + str(i)
            key12 = "rate" + str(i)
            key21 = "met" + str(i)
            key22 = "wspd" + str(i)
            key23 = "wdir" + str(i)
            key24 = "stab" + str(i)
            pass

        #Since the model runs very quickly, so the cgi-mode should work.
        model = model_puff_core(ReleaseQ, MetField, MetSeq)
        pass

    def GET(self):
        raise web.Forbidden()

if __name__ == "__main__":
    app.run()
