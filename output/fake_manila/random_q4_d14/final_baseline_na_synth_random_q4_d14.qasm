OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(5.8547128) q[1];
rz(5.5437815) q[2];
rz(1.8981271) q[3];
rz(1.6266203) q[4];
rz(4.3851436) q[4];
rz(1.3285095) q[4];
rz(3.2604824) q[4];
rz(0.89375868) q[4];
rz(0.71754831) q[4];
rz(2.7244279) q[4];
rz(1.7736761) q[4];
rz(0.80583618) q[4];
rz(5.9756145) q[4];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[3],q[2];
rz(3.4812765) q[2];
rz(6.2530914) q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[3],q[2];
rz(2.9545866) q[2];
rz(5.0496265) q[2];
cx q[1],q[2];
rz(5.949045) q[2];
rz(2.1844525) q[2];
rz(4.2567775) q[2];
rz(1.8526621) q[2];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
rz(2.4870577) q[1];
cx q[2],q[3];
rz(3.3281771) q[2];
rz(6.1712699) q[3];
rz(5.7423328) q[3];
rz(6.1469842) q[3];
rz(2.6288983) q[3];
rz(4.2160231) q[3];
