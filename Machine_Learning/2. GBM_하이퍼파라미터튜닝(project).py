import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# 전처리된 데이터 로딩
car = pd.read_csv('./car_price_remove_one_hot_encoding.csv')

# 입력(x), 출력(y) 분리
x = car.drop("price", axis = 1)
y = car["price"]

# 학습 데이터와 테스트 데이터로 분리 (7 : 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.70, random_state=1)

# 초기 GridSearch (파라미터 없음 - 기본 성능 확인용)
param_grid ={  } 
grid_search = GridSearchCV(GradientBoostingRegressor(loss= 'huber'),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(x_train, y_train)

# 테스트 데이터 예측
y_pred = grid_search.predict(x_test)

# 성능 출력
print('튜닝 전 훈련 세트의 정확도 : {:.4f} '. format(grid_search.score(x_train, y_train)))
print('튜닝 전 테스트 세트의 정확도 : {:.4f} '. format(grid_search.score(x_test, y_test)))

# 평균 절대 오차 출력
mae_pred =  mean_absolute_error(y_test, grid_search.predict(x_test))
print('튜닝 전 경사부스팅의 오차: {:.4f}'.format(mae_pred))

# 1차 튜닝: n_estimators만 조정
n_estimators = [100, 500, 1000]
param_grid = {'n_estimators' : n_estimators}

gb = GridSearchCV(GradientBoostingRegressor(loss= 'huber'), 
                  param_grid, cv=3, return_train_score=True)
gb.fit(x_train, y_train)

# 1차 튜닝 결과 출력
print('1차 튜닝 최적 하이퍼 파라미터: ', gb.best_params_)
print('1차 튜닝 훈련 세트의 정확도 : {:.4f} '. format(gb.score(x_train, y_train)))
print('1차 튜닝 테스트 세트의 정확도 : {:.4f} '.format(gb.score(x_test, y_test)))

mae_pred =  mean_absolute_error(y_test, gb.predict(x_test))
print('1차 튜닝 경사부스팅의 오차: {:.4f}'.format(mae_pred))

# 2차 튜닝: learning_rate 추가 조정
learning_rate = [0.1, 0.5, 1]

param_grid = {'n_estimators' : [500],
              'learning_rate' : learning_rate}

gb_2 = GridSearchCV(GradientBoostingRegressor(loss= 'huber'),
                  param_grid, cv=3, return_train_score=True)
gb_2.fit(x_train, y_train)

# 2차 튜닝 결과 출력
print('2차 튜닝 최적 하이퍼 파라미터: ', gb_2.best_params_)
print('2차 튜닝 훈련 세트의 정확도 : {:.4f} '. format(gb_2.score(x_train, y_train)))
print('2차 튜닝 테스트 세트의 정확도 : {:.4f} '.  format(gb_2.score(x_test, y_test))) 

mae_pred =  mean_absolute_error(y_test, gb_2.predict(x_test))
print('2차 튜닝 경사부스팅의 오차: {:.4f}'.format(mae_pred))

# 3차 튜닝: max_depth 추가
max_depth = [1, 2, 3, 4]

param_grid = {'n_estimators' : [500],
              'learning_rate' : [0.1],
              'max_depth' : max_depth}

gb_3 = GridSearchCV(GradientBoostingRegressor(loss= 'huber'),
                  param_grid, cv=3, return_train_score=True)
gb_3.fit(x_train, y_train)

# 3차 튜닝 결과 출력
print('3차 튜닝 최적 하이퍼 파라미터: ', gb_3.best_params_)
print('3차 튜닝 훈련 세트의 정확도 : {:.4f} '. format(gb_3.score(x_train, y_train)))
print('3차 튜닝 테스트 세트의 정확도 : {:.4f} '.  format(gb_3.score(x_test, y_test)))

mae_pred =  mean_absolute_error(y_test, gb_3.predict(x_test))
print('3차 튜닝 경사부스팅의 오차: {:.4f}'.format(mae_pred))

# 4차 튜닝: min_samples_leaf 추가
min_samples_leaf = [5, 10, 15]

param_grid = {'n_estimators' : [500],
              'learning_rate' : [0.1],
              'max_depth' : [2],
              'min_samples_leaf' : min_samples_leaf}

gb_4 = GridSearchCV(GradientBoostingRegressor(loss= 'huber'),
                  param_grid, cv=3, return_train_score=True)
gb_4.fit(x_train, y_train)

# 4차 튜닝 결과 출력
print('4차 튜닝 최적 하이퍼 파라미터: ', gb_4.best_params_)
print('4차 튜닝 훈련 세트의 정확도 : {:.4f} '. format(gb_4.score(x_train, y_train)))
print('4차 튜닝 테스트 세트의 정확도 : {:.4f} '. format(gb_4.score(x_test, y_test))) 

mae_pred =  mean_absolute_error(y_test, gb_4.predict(x_test))
print('4차 튜닝 경사부스팅의 오차: {:.4f}'.format(mae_pred))

# 5차 튜닝: min_samples_split 추가
min_samples_split = [2, 5, 10]

param_grid = {'n_estimators' : [500],
              'learning_rate' : [0.1],
              'max_depth' : [2],
              'min_samples_leaf' : [5],
              'min_samples_split' : min_samples_split}

gb_5 = GridSearchCV(GradientBoostingRegressor(loss= 'huber'),
                  param_grid, cv=3, return_train_score=True)
gb_5.fit(x_train, y_train)

# 5차 튜닝 결과 출력
print('5차 튜닝 최적 하이퍼 파라미터: ', gb_5.best_params_)
print('5차 튜닝 훈련 세트의 정확도 : {:.4f} '. format(gb_5.score(x_train, y_train)))
print('5차 튜닝 테스트 세트의 정확도 : {:.4f} '. format(gb_5.score(x_test, y_test)))

mae_pred =  mean_absolute_error(y_test, gb_5.predict(x_test))
print('5차 튜닝 경사부스팅의 오차: {:.4f}'.format(mae_pred))

# 최종 모델 단순화 (빠른 실행용)
param_grid = {'n_estimators' : [500],
              'learning_rate' : [0.1],
              'max_depth' : [2],
              'min_samples_leaf' : [5],
              'min_samples_split' : [10]}

gb_6 = GridSearchCV(GradientBoostingRegressor(loss= 'huber'),
                  param_grid, cv=3, return_train_score=True)
gb_6.fit(x_train, y_train)

# 최종 모델 결과 출력
print('최종 모델 최적 하이퍼 파라미터: ', gb_6.best_params_)
print('최종 모델 훈련 세트의 정확도 : {:.4f} '. format(gb_6.score(x_train, y_train)))
print('최종 모델 테스트 세트의 정확도 : {:.4f} '. format(gb_6.score(x_test, y_test))) 

mae_pred =  mean_absolute_error(y_test, gb_6.predict(x_test))
print('최종 모델 경사부스팅의 오차: {:.4f}'.format(mae_pred))

# 최종 모델 저장
joblib.dump( gb_6, 'car.model' )