## Cài đặt
1. Clone project về 
2. cd thư mục chưa project classifURL
3. build image docke với lệnh (docker build -t classify-url  . )
4. docker run -p 1995:1995 -v -i classify-url:latest


## Phân loai url với content

API phân loại dựa vào content

__Url:__ `/classify/url-with-content`

__Method:__ `POST`

__Auth required:__ `NO`

__Header:__

Key | Value
--- | ---
Content-type | application/json

__Request example:__

```bash
curl -X POST \
  http://0.0.0.0:1995/classify/url-with-content \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -H 'postman-token: 543b4768-ec96-0f02-e1f7-86bc5ff60c97' \
  -d '{
	"url": ["https://homedy.com/ban-san-thuong-mai-chung-cu-hyundai-hillstate-can-goc-120m2-gia-72-ty-es1166086?src=list_pc"]
}'
```

__Response example:__

```json
{
    "description": "Results are returned to you",
    "payload": [
        {
            "type": 1,
            "url": "https://homedy.com/ban-san-thuong-mai-chung-cu-hyundai-hillstate-can-goc-120m2-gia-72-ty-es1166086?src=list_pc"
        }
    ],
    "status": 0
}
```
Chạy lệnh tại máy 


1.  Lệnh phân loại:
*  python main-content.py --name content --predict "https://dothi.net/tin-thi-truong.htm, https://dothi.net/ban-can-ho-chung-cu-chung-cu-booyoung/ban-can-3-phong-ngu-9554m2-toa-chung-cu-booyoung-gia-255-ty-ban-cong-dong-nam-lh-0968103222-pr13066165.htm"
2. Lệnh train dữ liệu
*  python main-url.py --name content --train-model True --test-size 0.15

| Tham số  đầu vào |Giải thích|
| ------ | ------ |
| --name |tên model được config ở file conf.json |
| --predict |list các url cần phân loại dưới dạng string |
| --train-model |xác định train model hay không|
| --test-size |chia tập train và test theo tỉ lệ|
| --config | chỉ ra đường dẫn file config ví dụ như  file  trong thư mực config|

## Phân loại url không có content 

API phân loại không có content

__Url:__ `/classify-url/url-not-content`

__Method:__ `POST`

__Auth required:__ `NO`

__Header:__

Key | Value
--- | ---
Content-type | application/json

__Request example:__

```bash
curl -X POST \
  http://0.0.0.0:1995/classify-url/url-not-content \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -H 'postman-token: 94732909-9150-789e-307f-2f859bb1f9e1' \
  -d '{
	"domain": "dothi",
	"url": ["https://dothi.net/cho-thue-can-ho-chung-cu-vinhomes-dcapitale/chu-dau-tu-tan-hoang-minh-cho-thue-can-ho-vinhomes-dcapitale-1-pn-2pn-view-ho-full-noi-that-pr13066696.htm"]
}'
```

__Response example:__

```json
{
    "description": "Results are returned to you",
    "payload": [
        {
            "type": 1,
            "url": "https://dothi.net/cho-thue-can-ho-chung-cu-vinhomes-dcapitale/chu-dau-tu-tan-hoang-minh-cho-thue-can-ho-vinhomes-dcapitale-1-pn-2pn-view-ho-full-noi-that-pr13066696.htm"
        }
    ],
    "status": 0
}
```

Chạy lệnh tại máy 


1.  Lệnh phân loại:
*  python main-content.py --name dothi --predict "https://dothi.net/tin-thi-truong.htm, https://dothi.net/ban-can-ho-chung-cu-chung-cu-booyoung/ban-can-3-phong-ngu-9554m2-toa-chung-cu-booyoung-gia-255-ty-ban-cong-dong-nam-lh-0968103222-pr13066165.htm"
2. Lệnh train dữ liệu
*  python main-url.py --name batdongsan --train-model True --test-size 0.15

| Tham số  đầu vào |Giải thích|
| ------ | ------ |
| --name |tên model được config ở file conf.json |
| --predict |list các url cần phân loại dưới dạng string |
| --train-model |xác định train model hay không|
| --test-size |chia tập train và test theo tỉ lệ|
| --config | chỉ ra đường dẫn file config ví dụ như  file  trong thư mực config|
