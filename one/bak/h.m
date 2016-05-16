function prediction = h(x,theta)
prediction = x * theta';  % x는 m*(n+1) 행렬이어야함, theta는 (n+1) * 1행렬 이어야함
