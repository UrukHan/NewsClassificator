docker build -t bert .
docker run --rm --name rubert -p 8000:8000 bert
docker run -d --name rubert -p 8000:8000 bert

docker run --rm --gpus 'device=0' --name rubert -p 8000:8000 bert
docker run -it -d --gpus 'device=0' --name rubert -p 8000:8000 bert

/health_check
curl -X GET http://10.9.1.156:8000/api/health_check
curl -X POST -H "Content-Type: application/json" -d "{ \"text\": \"Введите текст\" }" http://192.168.1.25:8000/api/predict
curl -X POST -H "Content-Type: application/json" -d "{ \"text\": \"Введите текст\" }" http://172.20.192.1:8000/api/predict

curl -X POST -H "Content-Type: application/json" -d "{ \"text\": \"Администрация и прокуратура Кемерово организовали проверку достоверности опубликованной в сети видеозаписи того, как, по словам местных жителей, происходит захоронение бездомных на местном кладбище. Об этом в понедельник, 12 декабря, сообщает телеканал «360». В ролике с поднимающихся прицепов тракторов деревянные гробы сваливают в прорытые экскаватором траншеи. Некоторые из них разваливаются на части и видно, что внутри они пустые. Когда именно была сделана запись, не уточняется. Источники телеканала утверждают, что две местные ритуальные компании не поделили рынок и по итогам споров устроили демонстративную акцию со сбрасыванием пустых гробов в траншею. По другой версии, авторы ролика наглядно показали, каким образом МП «Спецбюро» хоронит бездомных после почти полного сокращения рабочих с 1 декабря 2016 года.\" }" localhost:8000/api/predict


docker exec -it <container_id> bash
tensorflow==2.8.0

//RUN apt-get install nvidia-container-runtime

curl -X POST -H "Content-Type: application/json" -d "{ \"text\": \"Украина по сути согласилась на принципиальные требования России, заявил глава российской делегации, помощник президента Владимир Мединский в эфире телеканала\" }" http://172.20.192.1:8000/api/predict


docker kill $(docker ps -a -q) 
docker rm $(docker ps -a -q)  
docker rmi $(docker images -a -q)

sudo chmod 666 /var/run/docker.sock

docker login http://docker-ici-cyrm.fdi.group:443
ici
cyrmE3r1x5a1J

docker tag news docker-ici-cyrm.fdi.group:443/class_news:310322
docker push docker-ici-cyrm.fdi.group:443/class_news:310322

docker tag t5 uruk/summarizet5:t5

