OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
cx q[5],q[4];
rz(5.7610013) q[1];
cx q[3],q[7];
rz(2.9204925) q[2];
cx q[6],q[0];
cx q[7],q[6];
cx q[3],q[2];
rz(2.4303559) q[1];
rz(1.2988238) q[0];
rz(3.286549) q[5];
rz(4.753567) q[4];
cx q[0],q[4];
rz(3.8698023) q[2];
cx q[6],q[3];
cx q[5],q[1];
rz(1.5447719) q[7];
rz(0.10637865) q[4];
cx q[1],q[0];
cx q[6],q[2];
cx q[5],q[3];
rz(4.2394642) q[7];
cx q[6],q[1];
cx q[3],q[2];
cx q[0],q[4];
rz(4.2767923) q[7];
rz(5.0958684) q[5];
rz(5.5061156) q[7];
rz(4.7397759) q[4];
cx q[5],q[0];
cx q[3],q[2];
cx q[1],q[6];
rz(5.9490644) q[5];
cx q[1],q[2];
rz(0.70892222) q[4];
cx q[7],q[6];
cx q[0],q[3];
rz(0.95147145) q[5];
cx q[3],q[1];
cx q[4],q[2];
rz(0.60625367) q[7];
cx q[6],q[0];
cx q[4],q[0];
rz(1.4461718) q[1];
rz(3.214208) q[5];
rz(0.56529836) q[3];
rz(3.8882347) q[7];
cx q[6],q[2];
rz(1.1484168) q[1];
cx q[0],q[4];
rz(0.75254039) q[5];
cx q[2],q[3];
cx q[6],q[7];
rz(1.4801627) q[7];
cx q[2],q[4];
rz(2.1542943) q[5];
rz(0.10834314) q[0];
rz(1.8336146) q[6];
rz(4.9656059) q[1];
rz(0.15291568) q[3];
