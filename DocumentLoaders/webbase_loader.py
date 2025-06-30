from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.flipkart.com/lenovo-60-45-cm-23-8-inch-full-hd-va-panel-3-side-near-edgeless-tuv-eye-care-monitor-d24-20-d24-40/p/itm7a2d267d21c3f?pid=MONFV5HRNF4QFVG4&lid=LSTMONFV5HRNF4QFVG4ATFEMN&marketplace=FLIPKART&store=6bo%2Fg0i%2F9no&srno=b_1_1&otracker=browse&otracker1=hp_rich_navigation_PINNED_neo%2Fmerchandising_NA_NAV_EXPANDABLE_navigationCard_cc_3_L2_view-all&fm=organic&iid=en_6bG7e3IUUrZTVqZkEhhBvGlidlMScino2XTglvBMo9n4pnTPEWhUVMRKAb2LoRznKsM2pDvrlnl-hijp2CA7MfUFjCTyOHoHZs-Z5_PS_w0%3D&ppt=hp&ppn=homepage&ssid=mf89hsok8g0000001750999943493")

docs = loader.load()
print(docs[0].page_content)