#ifndef _CLIENT_SERVER_H_
#define _CLIENT_SERVER_H_

class ClientServer
{
public:
    ClientServer();
    ~ClientServer();
    virtual void run() = 0;

protected:
    void transferClientToServer();
    void transferServerToClient();
};

#endif