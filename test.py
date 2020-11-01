from PIL import Image


path = '/data1/wangyanqing/data/webFG2020_sub/2473/2473_73070b806909ddeb06ed2790be543bbc6113d67b.jpg'

with open(path, 'rb') as f:
    img = Image.open(f)
    a = img.convert('RGB')

print(path)
print("success")