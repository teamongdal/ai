import os
import time
import requests
import pandas as pd
from tqdm import trange

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ===============================
# 1. Selenium 및 ChromeDriver 설정 (공용)
# ===============================
chrome_driver_path = r"C:\Users\015\musinsa\chromedriver-win64\chromedriver.exe"  # 본인 환경에 맞게 수정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 창 없이 실행 (디버깅 시 주석 처리 가능)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/98.0.4758.102 Safari/537.36")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
wait = WebDriverWait(driver, 15)

# ===============================
# Part A: 상품 링크 수집 (랭킹 페이지에서 최대 400개)
# ===============================
def scrape_product_links():
    df = pd.DataFrame()
    content_links = []
    page = 1
    # 400개 이상 링크를 수집하거나, 더 이상 상품이 없을 때까지 반복
    while len(content_links) < 400:
        url = f'https://www.musinsa.com/main/musinsa/ranking?storeCode=musinsa&sectionId=200&categoryCode=001000&contentsId=&page={page}'
        driver.get(url)
        time.sleep(3)  # 페이지 로드 대기
        products = driver.find_elements(By.CSS_SELECTOR, "a.sc-1m4cyao-2.bubXVJ.gtm-select-item")
        if not products:
            break
        for product in products:
            try:
                product_url = product.get_attribute("href")
                if product_url and "/products/" in product_url:
                    content_links.append(product_url)
            except Exception:
                continue
        print(f"페이지 {page}: {len(products)} 링크 수집 (누적 {len(content_links)}개)")
        page += 1
    # 최대 400개로 슬라이스
    content_links = content_links[:400]
    print(f"🔗 최종 수집된 상품 링크 개수: {len(content_links)}")
    df["내용링크"] = content_links
    return df

df_links = scrape_product_links()
df_links.to_csv('knit.csv', index=False)
print("✅ 상품 링크 수집 완료! → content_link.csv 저장됨")

# ===============================
# Part B: 상품 상세정보 및 리뷰 크롤링
# ===============================
# 이미지 저장 폴더 생성
save_folder = "knit"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def scrape_product_details(index, url):
    driver.get(url)
    time.sleep(3)
    
    # product_code: 폴더명 + "_" + (index+1) (0001부터 시작)
    product_code = f"{os.path.basename(save_folder)}_{index+1:04d}"
    
    try:
        product_name = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "span.text-lg.font-medium.break-all.flex-1.font-pretendard")
        )).text
    except Exception as e:
        print(f"[{url}] 상품명 추출 실패: {e}")
        product_name = "없음"
    
    try:
        original_price = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.sc-xz8kdb-0.drIrxb span[style*='line-through']")
        )).text
    except Exception as e:
        print(f"[{url}] 원가격 추출 실패: {e}")
        original_price = "없음"
    
    try:
        discount_rate = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.sc-xz8kdb-4.iccpET span.text-lg.font-semibold.mr-1.text-red")
        )).text
    except Exception as e:
        print(f"[{url}] 할인율 추출 실패: {e}")
        discount_rate = "없음"
    
    try:
        discounted_price = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.sc-xz8kdb-4.iccpET span.text-lg.font-semibold.text-black")
        )).text
    except Exception as e:
        print(f"[{url}] 할인가격 추출 실패: {e}")
        discounted_price = "없음"
    
    try:
        brand = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "a.gtm-click-brand span.text-sm.font-medium")
        )).text
    except Exception as e:
        print(f"[{url}] 브랜드 추출 실패: {e}")
        brand = "없음"
    
    try:
        category_elements = driver.find_elements(By.CSS_SELECTOR, "div.sc-147svlx-0.lneysx a.hTQFMT")
        category = " > ".join([elem.text.strip() for elem in category_elements if elem.text.strip() != ""])
        if not category:
            category = "없음"
    except Exception as e:
        print(f"[{url}] 카테고리 추출 실패: {e}")
        category = "없음"
    
    try:
        brand_img_elem = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.sc-11x022e-0.hzZrPp a.gtm-click-brand img")
        ))
        brand_img_url = brand_img_elem.get_attribute("src")
        if brand_img_url and not brand_img_url.startswith("http"):
            brand_img_url = "https:" + brand_img_url
    except Exception as e:
        print(f"[{url}] 브랜드 이미지 추출 실패: {e}")
        brand_img_url = "없음"
    
    # ----- 이미지 추출 (총 4개만 수집, product_images_5 제거)
    try:
        image_urls = []
        # 최대한 많이 스와이프하여 슬라이드 내의 모든 이미지를 확인
        swipe_attempts = 0
        max_swipes = 20  # 보수적으로 충분히 스와이프 시도
        while len(image_urls) < 4 and swipe_attempts < max_swipes:
            slide_elements = driver.find_elements(By.CSS_SELECTOR, "div.swiper-wrapper > div.swiper-slide")
            # 정렬은 선택사항; 필요시 data-swiper-slide-index 기준으로 정렬할 수 있음.
            for slide in slide_elements:
                try:
                    img_elem = slide.find_element(By.CSS_SELECTOR, "img.ljkzhU")
                    src = img_elem.get_attribute("src")
                    if src:
                        if not src.startswith("http"):
                            src = "https:" + src
                        if src not in image_urls:
                            response = requests.get(src)
                            if response.status_code == 200:
                                image_urls.append(src)
                                save_path = os.path.join(save_folder, f"{product_code}_{len(image_urls)-1:02d}.jpg")
                                with open(save_path, 'wb') as f:
                                    f.write(response.content)
                            else:
                                print(f"[{url}] 이미지 다운로드 실패 (HTTP {response.status_code}) - {src}")
                    if len(image_urls) >= 4:
                        break
                except Exception:
                    continue
            # 시도: "다음" 버튼이 있으면 클릭하여 슬라이드를 이동
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, ".swiper-button-next")
                next_btn.click()
                swipe_attempts += 1
                time.sleep(2)
            except Exception as e:
                print(f"[{url}] 다음 슬라이드 버튼 클릭 실패 또는 더 이상 슬라이드 없음: {e}")
                break
        while len(image_urls) < 4:
            image_urls.append("없음")
        image_urls_str = ", ".join(image_urls)
    except Exception as e:
        print(f"[{url}] 이미지 추출/다운로드 실패: {e}")
        image_urls_str = "없음"
    
    return product_name, original_price, discount_rate, discounted_price, brand, category, brand_img_url, image_urls_str, product_code

def extract_actual_review(full_text):
    """
    리뷰 전체 텍스트에서 실제 리뷰 내용만 추출합니다.
    (여기서는 단순히 줄 단위로 분리하여 '구매' 이후부터 숫자만 있는 줄 전까지의 내용을 연결)
    그리고 최종 결과의 끝에 ". 접기" 또는 "접기"가 있으면 제거합니다.
    """
    lines = full_text.splitlines()
    review_lines = []
    found_purchase = False
    for line in lines:
        stripped = line.strip()
        if not found_purchase:
            if "구매" in stripped:
                found_purchase = True
            continue
        else:
            if stripped.isdigit():
                break
            review_lines.append(stripped)
    if review_lines:
        candidate = " ".join(review_lines)
    else:
        candidate = ""
        for line in lines:
            if len(line.strip()) > len(candidate):
                candidate = line.strip()
    candidate = candidate.rstrip()
    if candidate.endswith(". 접기"):
        candidate = candidate[:-len(". 접기")].rstrip()
    elif candidate.endswith("접기"):
        candidate = candidate[:-len("접기")].rstrip()
    return candidate

def scrape_review_details(url):
    driver.get(url)
    time.sleep(2)
    try:
        like_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.sc-1wsabwr-2.kDMzAm.gtm-add-to-wishlist span.text-xs.font-medium.font-pretendard")
            )
        )
        like = like_element.text.strip()
        if like == "":
            like = "0"
    except Exception as e:
        print(f"[{url}] 좋아요 수 추출 실패: {e}")
        like = "없음"
    
    try:
        view_count = driver.find_element(By.XPATH, "//*[contains(text(),'조회수')]/following-sibling::*[1]").text
    except Exception as e:
        print(f"[{url}] 조회수 추출 실패: {e}")
        view_count = "없음"
    
    try:
        sales_count = driver.find_element(By.XPATH, "//*[contains(text(),'누적판매')]/following-sibling::*[1]").text
    except Exception as e:
        print(f"[{url}] 누적판매 추출 실패: {e}")
        sales_count = "없음"
    
    try:
        rating = driver.find_element(By.XPATH, "//div[@data-button-name='후기클릭']//span[contains(@class, 'font-pretendard') and contains(text(),'.')]").text
    except Exception as e:
        print(f"[{url}] 평점 추출 실패: {e}")
        rating = "없음"
    
    try:
        review_count = driver.find_element(By.XPATH, "//div[@data-button-name='후기클릭']//span[contains(text(),'후기')]").text
    except Exception as e:
        print(f"[{url}] 리뷰수 추출 실패: {e}")
        review_count = "없음"
    
    try:
        review_tab = driver.find_element(By.XPATH, "//div[@data-button-name='후기클릭']")
        driver.execute_script("arguments[0].click();", review_tab)
        time.sleep(2)
    except Exception as e:
        print(f"[{url}] 리뷰 탭 클릭 실패: {e}")
    
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)
    
    review_texts = ["없음"] * 5
    review_containers = driver.find_elements(By.CSS_SELECTOR, "div.review-list-item__Container-sc-13zantg-0.gtm-impression-content")
    if not review_containers:
        try:
            view_all_button = driver.find_element(By.XPATH, "//button[contains(text(),'전체보기')]")
            driver.execute_script("arguments[0].click();", view_all_button)
            time.sleep(2)
            review_containers = driver.find_elements(By.CSS_SELECTOR, "div.review-list-item__Container-sc-13zantg-0")
        except Exception as e:
            print(f"[{url}] 전체보기 버튼 클릭 실패 또는 리뷰 컨테이너 미발견: {e}")
    for j in range(5):
        if j < len(review_containers):
            try:
                container = review_containers[j]
                driver.execute_script('''
                    Array.from(arguments[0].querySelectorAll("span.text-body_13px_reg.underline.text-gray-600.font-pretendard"))
                    .forEach(el => {
                        if(el.innerText.trim() === "더보기" || el.innerText.trim() === "이전 후기 보기"){
                            el.remove();
                        }
                    });
                ''', container)
                try:
                    more_button = container.find_element(By.CSS_SELECTOR, "button.TruncateContent__MoreButton-sc-5tx4vi-3")
                    driver.execute_script("arguments[0].click();", more_button)
                    time.sleep(0.5)
                except Exception:
                    pass
                full_text = container.text
                actual_review = extract_actual_review(full_text)
                review_texts[j] = actual_review
            except Exception as inner_e:
                print(f"[{url}] 리뷰 {j+1} 추출 실패: {inner_e}")
                review_texts[j] = "없음"
        else:
            review_texts[j] = "없음"
    
    return like, view_count, sales_count, rating, review_count, review_texts

def process_category_split(cat_str):
    parts = [part.strip() for part in cat_str.split('>')]
    parts = [p for p in parts if not (p.startswith('(') and p.endswith(')'))]
    if len(parts) >= 2:
        return parts[0], parts[1]
    elif len(parts) == 1:
        return parts[0], ''
    else:
        return '', ''

# ===============================
# Part C: 메인 루프 - 상품 상세정보 및 리뷰 크롤링 (400개 상품)
# ===============================
df_links = pd.read_csv('content_link.csv')
top_n = min(400, len(df_links))
df_links = df_links.iloc[:top_n].copy()

final_data = []
for i in trange(top_n, desc="상품 상세 정보 및 리뷰 크롤링"):
    url = df_links['내용링크'].iloc[i]
    
    # (A) 상품 상세 정보 스크래핑
    (product_name, original_price, discount_rate, discounted_price,
     brand, category, brand_img_url, image_urls_str, product_code) = scrape_product_details(i, url)
    
    # (B) 리뷰 정보 스크래핑
    (like, view_count, sales_count, rating, review_count, review_texts) = scrape_review_details(url)
    
    # (C) 카테고리 전처리 (카테고리 1, 카테고리 2 분리)
    cat1, cat2 = process_category_split(category)
    
    # (D) 이미지 URL 분리: 최대 4개의 URL (없으면 "없음"으로 채움)
    image_urls_list = [x.strip() for x in image_urls_str.split(",")] if image_urls_str != "없음" else []
    while len(image_urls_list) < 4:
        image_urls_list.append("없음")
    image_urls_list = image_urls_list[:4]
    
    row = {
        "detail_url": url,
        "product_name": product_name,
        "product_price": original_price,
        "discount_rate": discount_rate,
        "final_price": discounted_price,
        "brand_name": brand,
        "brand_image": brand_img_url,
        "category": cat1,
        "category_sub": cat2,
        "product_images_1": image_urls_list[1],
        "product_images_2": image_urls_list[2],
        "product_images_3": image_urls_list[3],
        "product_images_4": image_urls_list[0],
        "product_code": product_code,
        "heart_cnt": like,
        "numof_views": view_count,
        "total_sales": sales_count,
        "review_cnt": review_count,
        "review_rating": rating,
        "review1": review_texts[0],
        "review2": review_texts[1],
        "review3": review_texts[2],
        "review4": review_texts[3],
        "review5": review_texts[4],
    }
    final_data.append(row)

final_columns = ["detail_url", "product_name", "product_price", "discount_rate", "final_price", "brand_name", "brand_image",
                 "category", "category_sub", "product_images_1", "product_images_2", "product_images_3", "product_images_4",
                 "product_code", "heart_cnt", "numof_views", "total_sales", "review_cnt", "review_rating", 
                 "review1", "review2", "review3", "review4", "review5"]

final_df = pd.DataFrame(final_data, columns=final_columns)
final_df.to_csv('final_musinsa_product_details.csv', index=False)
print("✅ 상품 상세 정보 및 리뷰 크롤링 완료! → final_musinsa_product_details.csv 저장됨")

driver.quit()