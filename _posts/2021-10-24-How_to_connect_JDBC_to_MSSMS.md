# How to connect JDBC to MSSMS (SQL server)

Their are 2 types of authentication in MSSMS
1) Windows authentication (we have this)

    No username or password required
    Hence,

    String connectionURL = "jdbc:sqlserver://localhost:1433;DatabaseName=project;integratedSecurity=true";

    contains no username and pasword, instead we add integratedSecurity=true;		

2) SQN server authentication

    Username and password are required.

    String connectionURL = "jdbc:sqlserver://localhost:1433;databaseName= project;user=<user>;password=<password>";

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
We aslo need sql server configuration manager. so to get this type mmc and open command.

1. This will open configuration manager
2. click on menu->add/remove snap in
3. sql server configuration manager-> add -> ok
4. menu-> save (save this file in a good location, desktop)
5. Now, this is your SQL configuration manager.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Now enable TCP/IP and Named pipes.
(https://stackoverflow.com/questions/33590030/the-tcp-ip-connection-to-the-host-localhost-port-1433-has-failed-error-need-as/33590981#33590981)

1. open SQL conf. manager.
2. go to sql server network configuration -> protocols for MSSQLserver
3. right click on tcp/ip enable it
4. right click on named pipes enable it.
5. All things are ready

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Add mssql-jdbc_auth.x.x.x64.dll to java bin forlder
(https://stackoverflow.com/questions/6087819/jdbc-sqlserverexception-this-driver-is-not-configured-for-integrated-authentic)

1. go to downloaded jdbc folder->enu->auth->x64 (according to your version)
2. copy the .dll file
3. paste this file to bin directory where your java is installed ( in my case D:\MARS Program Files\bin)
To know above location 
$ where javac

4. Now you are all good to go

-----------------------------------------------------------------------------------------------------------------------------------------------------

So using jdbc

You can directly make changes to database from jdbc programs.
