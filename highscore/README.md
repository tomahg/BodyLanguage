# Highscore for BodyFuck

A highscore server using node/express, accepting POST request from Python application and persisting a highscore list in json.\
Intended to be run locally only.

Start highscore server:
```
node server.js
```

## Testing from Postman
Verb: POST \
Url: http://127.0.0.1:3000/submit \
Body raw: 
``` json
{
  "time": 60
}
```