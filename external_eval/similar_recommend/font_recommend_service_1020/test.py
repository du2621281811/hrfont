'''
from ftplib import FTP

ftp = FTP()
ftp.connect(host='172.19.52.32', port=21)
ftp.login(user='ftpuser', passwd='12345678')

print("当前路径:", ftp.pwd())

files = ftp.nlst()#列出当前所有路径
ftp.cwd('/home/ftpuser/shared_data') #切换远程目录
print("当前路径:", ftp.pwd())
#with open('test_download.txt', 'wb') as f:
#    ftp.retrbinary('RETR test_upload.txt', f.write)
ftp.quit()
'''
from ftplib import FTP
# 启动ftp服务
ftp = FTP()
ftp.connect(host='172.19.52.32', port=21)
ftp.login(user='ftpuser', passwd='12345678')
ttf_path = 'font_files_new/20678@方正字迹-欧阳长迪行楷 简@方正字迹-欧阳长迪行楷@FZZJ-OYCDXKJW.TTF'
ttf_url = 'target.TTF'
#上传文件到ftp服务
with open(ttf_path, 'rb') as f:
    ftp.storbinary('STOR '+ ttf_url, f)
# 下载ttf文件
with open('font_files_new/target.TTF', 'wb') as f:
    ftp.retrbinary('RETR ' + ttf_url, f.write)    
files = ftp.nlst()
print(files)