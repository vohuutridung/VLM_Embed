mkdir -p vlm2vec_train/MMEB-train/images

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/ImageNet_1K.zip
unzip ImageNet_1K.zip -d ./vlm2vec_train/MMEB-train/images/
rm ImageNet_1K.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/N24News.zip
unzip N24News.zip -d ./vlm2vec_train/MMEB-train/images/
rm N24News.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/HatefulMemes.zip
unzip HatefulMemes.zip -d ./vlm2vec_train/MMEB-train/images/
rm HatefulMemes.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VOC2007.zip
unzip VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
rm VOC2007.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/SUN397.zip
unzip SUN397.zip -d ./vlm2vec_train/MMEB-train/images/
rm SUN397.zip