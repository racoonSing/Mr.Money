import pandas as pd
import matplotlib.pyplot as plt

# 여러 브랜드 데이터를 처리할 수 있도록 리스트 준비 (현재는 hyundai만 사용)
tmp = []
brands = ['hyundai']

# 브랜드별 파일 경로 지정
for brand in brands:
    file_path = f"./{brand}.csv"
    df = pd.read_csv(file_path)

    # brand 컬럼 추가
    df["brand"] = brand

    tmp.append(df)

# 전체 병합
df_total = pd.concat(tmp, ignore_index=True)

# 중복 제거 (모델명, 연식, 오일종류, 주행거리, 가격이 모두 같은 행 제거)
df_total = df_total.drop_duplicates(
    subset=["model", "year", "oilingtype", "mileage", "price"]
).reset_index(drop=True)

# 가격 기준 이상치 제거: 500만원 초과, 4000만원 미만
condition = (df_total["price"] < 4000) & (df_total["price"] > 500)
df_total_price_remove = df_total[condition]

# 박스플롯을 통해 이상치 제거 시각적 확인
boxplot = df_total_price_remove.boxplot(column=['price'])
boxplot.plot()
plt.title('price Boxplot')
plt.show()

# 주행거리(mileage)를 10,000km 단위로 구간화하여 컬럼 생성 (one-hot 인코딩 준비)
bins = list(range(0, 200001, 10000)) # 0 ~ 200,000km 구간

for lower, upper in zip(bins[:-1], bins[1:]):
    if lower == 0:
        cond = df_total_price_remove['mileage'].between(lower, upper)
        col_name = f"mileage_0_{upper}"
    else:
        cond = df_total_price_remove['mileage'].gt(lower) & df_total_price_remove['mileage'].le(upper)
        col_name = f"mileage_{lower+1}_{upper}"
    df_total_price_remove[col_name] = cond
   
# 원래의 mileage 컬럼 제거 (이제 구간화된 컬럼들로 대체됨)
df_total_price_remove = df_total_price_remove.drop(columns=['mileage'])

# year 컬럼을 문자열로 변환 (one-hot 인코딩 시 숫자로 처리되지 않도록)
df_total_price_remove.year = df_total_price_remove.year.astype('str')

# 원핫인코딩 진행
df_total_price_remove = pd.get_dummies(df_total_price_remove, prefix_sep='_', drop_first=False)

# 최종 전처리 결과 저장 (머신러닝 모델 입력용)
df_total_price_remove.to_csv("./car_price_remove_one_hot_encoding.csv", index=False)