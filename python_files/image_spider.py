
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup as bs
import requests


# In[ ]:


import re  
import requests  

def dowmloadPic(html,keyword , i ):  
    pic_url = re.findall('"objURL":"(.*?)",',html,re.S)     
    print '找到关键词:'+keyword+'的图片，现在开始下载图片...'  
    for each in pic_url:  
        print u'正在下载第'+str(i+1)+u'张图片，图片地址:'+str(each)  
        try:  
            pic= requests.get(each, timeout=50)  
        except  Exception,ex :  
            print u'【错误】当前图片无法下载'   
            continue  
        string = 'pictures\\'+keyword+'_'+str(i) + '.jpg'  
        #resolve the problem of encode, make sure that chinese name could be store  
        fp = open(string.decode('utf-8').encode('cp936'),'wb')  
        fp.write(pic.content)  
        fp.close()  
        i += 1  
    return i  
  
if __name__ == '__main__':  
    word =  raw_input('Input keywords:')  
    word = word.decode('cp936').encode('utf-8')  
    #url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&ct=201326592&v=flip'  
    pnMax = input('Input max pn:')  
    pncount = 0  
    gsm = 80  #这个值不知干嘛的  
    str_gsm =str(gsm)  
    if not os.path.exists('pictures'):  
        os.mkdir('pictures')  
    while pncount<pnMax:  
        str_pn = str(pncount)  
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&pn='+str_pn+'&gsm='+str_gsm+'&ct=&ic=0&lm=-1&width=0&height=0'  
        result = requests.get(url)  
        pncount = dowmloadPic(result.text,word ,pncount)  
    print u'下载完毕'  

