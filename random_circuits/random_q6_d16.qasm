OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
rz(1.6649882) q[3];
rz(3.331733) q[0];
rz(1.5947944) q[5];
rz(1.11595) q[1];
cx q[2],q[4];
cx q[3],q[0];
cx q[5],q[4];
cx q[2],q[1];
rz(0.90942233) q[2];
rz(4.9812483) q[1];
cx q[5],q[0];
cx q[3],q[4];
cx q[2],q[4];
cx q[1],q[5];
rz(6.2151311) q[3];
rz(5.8292624) q[0];
cx q[4],q[2];
rz(6.2699148) q[5];
cx q[3],q[0];
rz(5.8170694) q[1];
cx q[1],q[3];
rz(5.4977982) q[5];
cx q[0],q[2];
rz(3.0601474) q[4];
cx q[1],q[3];
rz(4.2691825) q[4];
rz(4.2754424) q[2];
rz(3.5699369) q[5];
rz(4.6766377) q[0];
cx q[0],q[5];
cx q[2],q[4];
rz(5.3655464) q[1];
rz(6.133985) q[3];
rz(2.3421374) q[4];
cx q[2],q[3];
cx q[5],q[0];
rz(2.0860808) q[1];
rz(4.0519505) q[3];
rz(5.938491) q[1];
rz(3.4921624) q[2];
cx q[4],q[0];
rz(0.35718939) q[5];
rz(1.5212854) q[3];
rz(5.6463228) q[4];
rz(2.9026761) q[1];
rz(0.20400415) q[0];
cx q[2],q[5];
cx q[4],q[0];
rz(2.0189476) q[1];
rz(4.9972746) q[5];
cx q[3],q[2];
cx q[4],q[3];
cx q[5],q[2];
cx q[1],q[0];
cx q[1],q[4];
rz(1.2187848) q[5];
cx q[0],q[3];
rz(5.2022516) q[2];
rz(6.0083432) q[1];
cx q[2],q[3];
cx q[5],q[4];
rz(6.259547) q[0];
rz(0.15039756) q[1];
rz(2.1328518) q[3];
rz(2.6440639) q[2];
cx q[5],q[4];
rz(4.8321172) q[0];
