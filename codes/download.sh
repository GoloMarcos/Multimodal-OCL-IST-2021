#download Datasets
[ ! -f "datasets/ARE.plk" ] && echo "Downloading ARE" && gdown --id link_to_ARE -O "datasets/ARE.plk"

[ ! -f "datasets/TIN.plk" ] && echo "Downloading TIN" && gdown --id link_to_TIN -O "datasets/TIN.plk"

[ ! -f "datasets/TEN.plk" ] && echo "Downloading TEN" && gdown --id link_to_TEN -O "datasets/TEN.plk"

# These datasets are private and may only be used for academic purposes. We got the link to the datasets by contacting their owners. So, if you want the link to the datasets, please contact them.
