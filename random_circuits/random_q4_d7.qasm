OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(4.8301736) q[3];
rz(2.6887607) q[2];
rz(3.7314528) q[0];
rz(5.8778444) q[1];
cx q[2],q[0];
cx q[1],q[3];
rz(5.2069133) q[1];
cx q[0],q[2];
rz(5.7885974) q[3];
rz(6.0478022) q[2];
rz(4.8715763) q[0];
cx q[1],q[3];
rz(0.1117334) q[0];
rz(1.054128) q[2];
cx q[3],q[1];
rz(4.5994311) q[3];
rz(4.1836129) q[2];
rz(4.6360688) q[0];
rz(0.20270951) q[1];
cx q[1],q[0];
rz(3.7886901) q[3];
rz(1.0520249) q[2];
