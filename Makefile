days=$(date -v -365d '+%Y%m%d')
get-days: $(days)
	@echo $(days)

download-data:
	@echo "Downloading Data..."
	docker-compose run --rm freqtrade download-data --config user_data/config.json --timeframes 1h --timerange 20230101-

optimize-all:
	@echo "Optimizing All Spaces..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 3000 --timerange 20230101-20230902 --timeframe 1h --spaces all

optimize-default:
	@echo "Optimizing Default Spaces..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 5000 --timerange 20230101-20230902 --timeframe 1h --spaces default

optimize-buy:
	@echo "Optimizing Buy Space..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 2000 --timerange 20230101-20230902 --timeframe 1h --spaces buy

optimize-sell:
	@echo "Optimizing Sell Space..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 2000 --timerange 20230101-20230902 --timeframe 1h --spaces sell

optimize-roi:
	@echo "Optimizing ROI Space..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 2000 --timerange 20230101-20230902 --timeframe 1h --spaces roi

optimize-stoploss:
	@echo "Optimizing Stoploss Space..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 300 --timerange 20230101-20230902 --timeframe 1h --spaces stoploss

optimize-trailing:
	@echo "Optimizing Trailing Space..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 300 --timerange 20230101-20230902 --timeframe 1h --spaces trailing

optimize-trades:
	@echo "Optimizing Trades Space..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 100 --timerange 20230101-20230902 --timeframe 1h --spaces trades

optimize-protection:
	@echo "Optimizing Protection Space..."
	docker-compose run --rm freqtrade hyperopt --hyperopt-loss CombinedProfitLoss --strategy AwesomeStrategy --config user_data/config.json -e 300 --timerange 20230101-20230902 --timeframe 1h --spaces protection

backtest:
	@echo "Conducting Backtest..."
	docker-compose run --rm freqtrade backtesting --strategy AwesomeStrategy --dry-run-wallet 1500 --timerange 20230903-

show-backtest-results:
	@echo "Show Backtest Results..."
	docker-compose run --rm freqtrade backtesting-show --show-pair-list

test-pairlist:
	@echo "Test Pairlist..."
	docker-compose run --rm freqtrade test-pairlist --config user_data/config-with-dynamic-pairlist.json --quote USDT

optimize-spaces:
	make optimize-buy & make optimize-sell & make optimize-roi & make optimize-stoploss & make optimize-trailing & make optimize-trades & make optimize-protection

copy_config:
	scp -i "freqtrade-ec2-access.pem" ./freqtrade-2023-git/user_data/config.json ubuntu@ec2-{ec2-ip}.compute-1.amazonaws.com:/home/ubuntu/freqtrade-2023/user_data/config.json 