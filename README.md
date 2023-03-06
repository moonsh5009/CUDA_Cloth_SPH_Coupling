# CUDA_Cloth_SPH_Coupling

논문 : https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11213514
 - 다공성 물질의 물리적 특성을 이용한 SPH와 Cloth Dynamics의 상호작용
 - 대학교 학부연구동아리에서 진행한 프로젝트들을 커플링한 프로젝트

SPH는 DFSPH기법으로 구현

SurfaceTension은 Akinci기법으로 구현

ClothDynamics는 Projective Dynamics기법을 Chebyshev Semi-Iterative기법으로 가속화 하여 구현

Particle Sampling은 Adams기법으로 구현

Particle과 Triangle Mesh사이 충돌은 Bridson논문을 참조하여 구현


참고문헌
 - Adams, Bart, Mark Pauly, Richard Keiser, and Leonidas J. Guibas. "Adaptively sampled particle fluids." In ACM SIGGRAPH 2007 papers, pp. 48-es. 2007.
 - Akinci, Nadir, Jens Cornelis, Gizem Akinci, and Matthias Teschner. "Coupling elastic solids with smoothed particle hydrodynamics fluids." Computer Animation and Virtual Worlds 24, no. 3-4 (2013): 195-203.
 - Akinci, Nadir, Markus Ihmsen, Gizem Akinci, Barbara Solenthaler, and Matthias Teschner. "Versatile rigid-fluid coupling for incompressible SPH." ACM Transactions on Graphics (TOG) 31, no. 4 (2012): 1-8.
 - Akinci, Nadir, Gizem Akinci, and Matthias Teschner. "Versatile surface tension and adhesion for SPH fluids." ACM Transactions on Graphics (TOG) 32, no. 6 (2013): 1-8.
 - Bridson, Robert, Ronald Fedkiw, and John Anderson. "Robust treatment of collisions, contact and friction for cloth animation." In Proceedings of the 29th annual conference on Computer graphics and interactive techniques, pp. 594-603. 2002
 - Bender, J., C. Duriez, F. Jaillet, and G. Zachmann. "Coupling Hair with Smoothed Particle Hydrodynamics Fluids."
 - Solenthaler, Barbara, and Markus Gross. "Two-scale particle simulation." In ACM SIGGRAPH 2011 papers, pp. 1-8. 2011
 - H. Wang. A chebyshev semi-iterative approach for accelerating projective and position-based dynamics. ACM Trans. Graph., vol. 34, no. 6, pp. 246:1–246:9, Oct. 2015.
