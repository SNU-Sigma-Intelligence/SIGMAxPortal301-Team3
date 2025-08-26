1. pose_test(single_version_fast).py를 사용하기 전, 14~18번 줄의 port, baudrate를 수정한다. 또한, max_len(측정할 데이터 수)와 sensor_range(테스트 할 거리)를 입력한다. version은 실험에 따라 바꾼다.
2. pose_test.(single_version_fast).py는 완료시 sensor_data(all).xlsx에 데이터를 자동으로 저장하므로 잘못 측정한 데이터가 있을 시 반드시!!! 다시 측정하기 전에 xlsx파일에 들어가서 잘못 측정한 데이터를 지우고 실험해야 한다.
3. regression_analysis.py는 지금까지 sensor_data(all).xlsx에 저장된 데이터를 기반으로 통계 분석을 해주는 코드이다.
