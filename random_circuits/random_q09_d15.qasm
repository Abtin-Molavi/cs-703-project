OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[8],q[4];
cx q[5],q[6];
rz(3.9772914) q[3];
rz(2.382973) q[1];
cx q[2],q[7];
rz(3.3219244) q[0];
cx q[1],q[8];
cx q[0],q[6];
cx q[4],q[3];
cx q[7],q[2];
rz(3.0361309) q[5];
rz(3.2715791) q[1];
cx q[5],q[0];
cx q[3],q[6];
cx q[8],q[4];
rz(1.1626409) q[2];
rz(6.1699702) q[7];
rz(4.821722) q[0];
cx q[5],q[8];
cx q[4],q[7];
cx q[2],q[1];
rz(5.2754585) q[3];
rz(0.99865695) q[6];
rz(0.5485377) q[3];
cx q[8],q[5];
cx q[7],q[0];
rz(4.0132096) q[4];
cx q[6],q[1];
rz(6.1980251) q[2];
rz(3.6385343) q[1];
cx q[3],q[6];
cx q[4],q[5];
rz(3.5834545) q[7];
cx q[8],q[2];
rz(6.1863485) q[0];
cx q[6],q[5];
rz(2.9210051) q[8];
cx q[0],q[7];
rz(1.8125195) q[2];
rz(6.2330221) q[4];
rz(1.0475726) q[1];
rz(6.1755529) q[3];
rz(0.071880452) q[1];
cx q[7],q[5];
cx q[0],q[4];
rz(3.6639362) q[8];
cx q[3],q[2];
rz(3.6199729) q[6];
rz(2.8615122) q[0];
cx q[4],q[8];
rz(0.74521361) q[5];
cx q[2],q[3];
cx q[1],q[7];
rz(1.8208153) q[6];
cx q[7],q[6];
cx q[1],q[2];
cx q[5],q[0];
cx q[3],q[8];
rz(3.4751196) q[4];
cx q[0],q[3];
rz(4.7238978) q[1];
cx q[5],q[4];
cx q[8],q[6];
rz(3.2338498) q[7];
rz(3.348686) q[2];
rz(2.3991098) q[2];
cx q[7],q[4];
rz(1.7714279) q[0];
rz(3.400598) q[1];
rz(1.192588) q[6];
cx q[8],q[3];
rz(4.9455748) q[5];
cx q[6],q[3];
rz(0.095707142) q[7];
cx q[5],q[4];
cx q[2],q[0];
rz(1.3546662) q[8];
rz(0.11994795) q[1];
rz(0.56089451) q[6];
cx q[7],q[2];
rz(5.6037908) q[0];
rz(4.5093453) q[8];
rz(2.3048055) q[1];
rz(1.7833914) q[3];
rz(6.2738429) q[5];
rz(4.4457821) q[4];
cx q[3],q[5];
rz(2.745601) q[0];
rz(6.1908289) q[7];
rz(4.0196556) q[6];
cx q[8],q[2];
cx q[1],q[4];