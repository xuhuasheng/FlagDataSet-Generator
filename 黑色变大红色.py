from PIL import Image

i = 1
j = 1
img = Image.open("./1.png")#读取系统的内照片
print (img.size)#打印图片大小
print (img.getpixel((4,4)))

width = img.size[0]#长度
height = img.size[1]#宽度
for i in range(0,width):#遍历所有长度的点
    for j in range(0,height):#遍历所有宽度的点
        data = (img.getpixel((i,j)))#打印该图片的所有点
        print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
        print (data[0])#打印RGBA的r值
        if (data[0]<10 and data[1]<=10 and data[2]<=100):#RGBA
            img.putpixel((i,j),(234,53,57,255))#则这些像素点的颜色改成大红色
img = img.convert("RGB")#把图片强制转成RGB
img.save("./1.jpg")#保存修改像素点后的图片
