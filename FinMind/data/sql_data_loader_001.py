# -*- coding: utf-8 -*-
import typing
from typing import Union, Optional
import sqlite3
import pandas as pd
from FinMind.schema.data import Dataset

# 這個類別用 SQLite 當資料來源；介面模擬 FinMind.DataLoader
class DataLoader:
    def __init__(self, db_path: Union[str, sqlite3.Connection] = "stock.db"):
        if isinstance(db_path, str):
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        else:
            self.conn = db_path
        # 與原版對齊（若外部有定義 Feature 再掛上去，不然就先 None）
        try:
            self.feature = Feature(self)  # noqa: F821
        except NameError:
            self.feature = None

    # --------- 小工具 ---------
    def _table_exists(self, name: str) -> bool:
        sql = "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?"
        return pd.read_sql_query(sql, self.conn, params=[name]).shape[0] > 0

    def _prefer_view(self, view_name: str, table_name: str) -> str:
        return view_name if self._table_exists(view_name) else table_name

    # --------- FinMind 標準入口：get_data ---------
    def get_data(
        self,
        dataset: typing.Union[Dataset, str],
        data_id: str = "",
        start_date: str = "",
        end_date: str = "",
        timeout: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        key = dataset.value if isinstance(dataset, Dataset) else str(dataset)

        # 0) 台股總覽 TaiwanStockInfo
        if key == "TaiwanStockInfo":
            if not self._table_exists("tw_stock_info"):
                return pd.DataFrame(columns=["industry_category","stock_id","stock_name","type"])
            q = "SELECT industry_category, stock_id, stock_name, type FROM tw_stock_info"
            return pd.read_sql_query(q, self.conn)


        # 1) 台股總覽(含權證) TaiwanStockInfoWithWarrant
        if key == "TaiwanStockInfoWithWarrant":
            if not self._table_exists("tw_stock_info_warrant"):
                return pd.DataFrame(columns=["industry_category","stock_id","stock_name","type"])
            q = "SELECT industry_category, stock_id, stock_name, type FROM tw_stock_info_warrant"
            return pd.read_sql_query(q, self.conn)

        # 2) 台股總覽(含權證) TaiwanSecuritiesTraderInfo
        if key == "TaiwanSecuritiesTraderInfo":
            if not self._table_exists("tw_securities_trader_info"):
                return pd.DataFrame(columns=["securities_trader_id","securities_trader","date","address","phone"])
            q = "SELECT securities_trader_id, securities_trader, date, address, phone FROM tw_stock_info_warrant"
            return pd.read_sql_query(q, self.conn)


        # …… get_data 開頭的其它分支之後
        if key == "TaiwanStockMarginPurchaseShortSale":
            cols = [
                "date",
                "stock_id",
                "MarginPurchaseBuy",
                "MarginPurchaseCashRepayment",
                "MarginPurchaseLimit",
                "MarginPurchaseSell",
                "MarginPurchaseTodayBalance",
                "MarginPurchaseYesterdayBalance",
                "Note",
                "OffsetLoanAndShort",
                "ShortSaleBuy",
                "ShortSaleCashRepayment",
                "ShortSaleLimit",
                "ShortSaleSell",
                "ShortSaleTodayBalance",
                "ShortSaleYesterdayBalance",
            ]
            if not self._table_exists("tw_margin_purchase_short_sale"):
                # 表不存在就回傳空的欄位結構
                return pd.DataFrame(columns=cols)

            # 基本查詢
            q = f"""
                SELECT {", ".join(cols)}
                FROM tw_margin_purchase_short_sale
                WHERE 1=1
            """
            params = []

            # 依需要加上 stock_id / 日期區間條件
            if data_id:
                q += " AND stock_id = ?"
                params.append(data_id)

            if start_date:
                q += " AND date >= ?"
                params.append(start_date)

            if end_date:
                q += " AND date <= ?"
                params.append(end_date)

            q += " ORDER BY date ASC"

            return pd.read_sql_query(q, self.conn, params=params)




        # 1) 台股日價 TaiwanStockPrice
        if key == "TaiwanStockPrice":
            # 特例：大盤比較（TAIEX）
            # if data_id == "TAIEX":
            #     if self._table_exists("tw_index_price"):
            #         q = """
            #             SELECT date, close FROM tw_index_price
            #             WHERE (?='' OR date>=?)
            #               AND (?='' OR date<=?)
            #             ORDER BY date
            #         """
            #         df = pd.read_sql_query(q, self.conn, params=[start_date, start_date, end_date, end_date])
            #         df["stock_id"] = "TAIEX"
            #         return df[["date", "stock_id", "close"]]
            #     return pd.DataFrame(columns=["date", "stock_id", "close"])

            src = self._prefer_view("vw_price_daily", "tw_price_daily")
            if src == "vw_price_daily":
                q = """
                    SELECT date, stock_id,
                           open,
                           high   AS max,
                           low    AS min,
                           close,
                           volume AS Trading_Volume,
                           turnover AS Trading_money,
                           spread,
                           trade_count AS Trading_turnover
                    FROM vw_price_daily
                    WHERE stock_id = ?
                      AND (?='' OR date>=?)
                      AND (?='' OR date<=?)
                    ORDER BY date
                """
            else:
                q = """
                    SELECT date, stock_id,
                           open, max, min, close,
                           Trading_Volume, Trading_money, spread, Trading_turnover
                    FROM tw_price_daily
                    WHERE stock_id = ?
                      AND (?='' OR date>=?)
                      AND (?='' OR date<=?)
                    ORDER BY date
                """
            df = pd.read_sql_query(q, self.conn, params=[data_id, start_date, start_date, end_date, end_date])
            need = ["date","stock_id","Trading_Volume","Trading_money","open","max","min","close","spread","Trading_turnover"]
            for c in need:
                if c not in df.columns:
                    df[c] = 0
            return df[need]

        # 2) 配息 TaiwanStockDividend
        if key == "TaiwanStockDividend":
            if not self._table_exists("tw_dividend"):
                return pd.DataFrame(columns=[
                    "stock_id",
                    "CashExDividendTradingDate","CashEarningsDistribution",
                    "StockExDividendTradingDate","StockEarningsDistribution",
                ])
            q = """
                SELECT stock_id,
                       CashExDividendTradingDate,
                       CashEarningsDistribution,
                       StockExDividendTradingDate,
                       StockEarningsDistribution
                FROM tw_dividend
                WHERE stock_id = ?
                  AND (
                        (?='' OR CashExDividendTradingDate>=?)
                     OR (?='' OR StockExDividendTradingDate>=?)
                  )
                  AND (
                        (?='' OR CashExDividendTradingDate<=?)
                     OR (?='' OR StockExDividendTradingDate<=?)
                  )
                ORDER BY COALESCE(CashExDividendTradingDate, StockExDividendTradingDate)
            """
            return pd.read_sql_query(
                q, self.conn,
                params=[data_id, start_date, start_date, start_date, start_date, end_date, end_date, end_date, end_date]
            )

        # 3) 三大法人明細 TaiwanStockInstitutionalInvestorsBuySell
        if key == "TaiwanStockInstitutionalInvestorsBuySell":
            src = self._prefer_view("vw_inst_flow_detail", "tw_inst_flow_detail")
            if not self._table_exists(src):
                return pd.DataFrame(columns=["date","stock_id","name","buy","sell","net"])
            q = f"""
                SELECT date, stock_id, name, buy, sell,
                       COALESCE(net, buy - sell) AS net
                FROM {src}
                WHERE stock_id = ?
                  AND (?='' OR date>=?)
                  AND (?='' OR date<=?)
                ORDER BY date, name
            """
            return pd.read_sql_query(q, self.conn, params=[data_id, start_date, start_date, end_date, end_date])

        # 4) 三大法人總表（如果有人直接要 Dataset.TaiwanStockTotalInstitutionalInvestors）
        if key == "TaiwanStockTotalInstitutionalInvestors":
            src = self._prefer_view("vw_inst_flow", "tw_inst_flow")
            if not self._table_exists(src):
                return pd.DataFrame(columns=["date","stock_id","buy","sell","net"])
            q = f"""
                SELECT date, stock_id, buy, sell, net
                FROM {src}
                WHERE (?='' OR date>=?)
                  AND (?='' OR date<=?)
                ORDER BY date
            """
            return pd.read_sql_query(q, self.conn, params=[start_date, start_date, end_date, end_date])

        # 5) 月營收 TaiwanStockMonthRevenue
        if key == "TaiwanStockMonthRevenue":
            src = self._prefer_view("vw_month_revenue", "tw_month_revenue")
            if not self._table_exists(src):
                return pd.DataFrame(columns=["stock_id","date","revenue"])
            q = f"""
                SELECT stock_id, date, revenue
                FROM {src}
                WHERE stock_id = ?
                  AND (?='' OR date>=?)
                  AND (?='' OR date<=?)
                ORDER BY date
            """
            return pd.read_sql_query(q, self.conn, params=[data_id, start_date, start_date, end_date, end_date])

        # 6) 財報 TaiwanStockFinancialStatements
        if key == "TaiwanStockFinancialStatements":
            # 你的 DB 是 wide：tw_fin_statement(eps, gross_profit, income_after_taxes, cogs, equity)
            # 轉回 FinMind long 格式：('type','value','origin_name')
            if not self._table_exists("tw_fin_statement"):
                return pd.DataFrame(columns=["date","stock_id","type","value","origin_name"])
            q = """
                SELECT stock_id, date, eps, gross_profit, income_after_taxes, cogs, equity
                FROM tw_fin_statement
                WHERE stock_id = ?
                  AND (?='' OR date>=?)
                  AND (?='' OR date<=?)
                ORDER BY date
            """
            wide = pd.read_sql_query(q, self.conn, params=[data_id, start_date, start_date, end_date, end_date])
            if wide.empty:
                return pd.DataFrame(columns=["date","stock_id","type","value","origin_name"])
            # map 成 FinMind 的英文科目名稱
            map_to_type = {
                "eps": "EPS",
                "gross_profit": "GrossProfit",
                "income_after_taxes": "IncomeAfterTaxes",
                "cogs": "CostOfGoodsSold",
                "equity": "EquityAttributableToOwnersOfParent",
            }
            origin_zh = {
                "EPS": "基本每股盈餘（元）",
                "GrossProfit": "營業毛利（毛損）淨額",
                "IncomeAfterTaxes": "本期淨利（淨損）",
                "CostOfGoodsSold": "營業成本",
                "EquityAttributableToOwnersOfParent": "綜合損益總額歸屬於母公司業主",
            }
            melt = wide.melt(id_vars=["stock_id","date"], var_name="col", value_name="value")
            melt["type"] = melt["col"].map(map_to_type)
            melt = melt.dropna(subset=["type"]).drop(columns=["col"])
            melt["origin_name"] = melt["type"].map(origin_zh)
            # 排序、欄位順序
            melt = melt[["date","stock_id","type","value","origin_name"]].sort_values(["date","type"])
            return melt.reset_index(drop=True)

        # 7) 台股總覽 TaiwanStockInfo（BackTest 起始會用到）


        # 8) 還原股價（若無資料，即回傳與原價相同欄位結構）
        if key == "TaiwanStockPriceAdj":
            df = self.get_data(Dataset.TaiwanStockPrice, data_id=data_id, start_date=start_date, end_date=end_date)
            return df

        # 其它 dataset 先回傳空表（盡量給出「正確欄名」以避免下游 KeyError）
        EMPTY_COLUMNS = {
            "TaiwanStockPER": ["date","stock_id","dividend_yield","PER","PBR"],
            "TaiwanStockStatisticsOfOrderBookAndTrade": ["date","Time","TotalBuyOrder","TotalBuyVolume","TotalSellOrder","TotalSellVolume","TotalDealOrder","TotalDealVolume","TotalDealMoney"],
            "TaiwanStockDayTrading": ["stock_id","date","BuyAfterSale","Volume","BuyAmount","SellAmount"],
            "TaiwanStockShareholding": ["date","stock_id","stock_name","InternationalCode","ForeignInvestmentRemainingShares","ForeignInvestmentShares","ForeignInvestmentRemainRatio","ForeignInvestmentSharesRatio","ForeignInvestmentUpperLimitRatio","ChineseInvestmentUpperLimitRatio","NumberOfSharesIssued","RecentlyDeclareDate","note"],
            "TaiwanStockHoldingSharesPer": ["date","stock_id","HoldingSharesLevel","people","percent","unit"],
            "TaiwanDailyShortSaleBalances": ["date","stock_id","MarginShortSalesPreviousDayBalance","MarginShortSalesShortSales","MarginShortSalesShortCovering","MarginShortSalesStockRedemption","MarginShortSalesCurrentDayBalance","MarginShortSalesQuota","SBLShortSalesPreviousDayBalance","SBLShortSalesShortSales","SBLShortSalesReturns","SBLShortSalesAdjustments","SBLShortSalesCurrentDayBalance","SBLShortSalesQuota","SBLShortSalesShortCovering"],
            "TaiwanStockCashFlowsStatement": ["date","stock_id","type","value","origin_name"],
            "TaiwanStockBalanceSheet": ["date","stock_id","type","value","origin_name"],
            "TaiwanStockDividendResult": ["date","stock_id","before_price","after_price","stock_and_cache_dividend","stock_or_cache_dividend","max_price","min_price","open_price","reference_price"],
            "TaiwanStockWeekPrice": ["stock_id","yweek","max","min","trading_volume","trading_money","trading_turnover","date","close","open","spread"],
            "TaiwanStockMonthPrice": ["stock_id","ymonth","max","min","trading_volume","trading_money","trading_turnover","date","close","open","spread"],
            "TaiwanStockMarketValue": ["date","stock_id","market_value"],
            "TaiwanStock10Year": ["date","stock_id","close"],
            "TaiwanStockDelisting": ["date","stock_id","stock_name"],
            "TaiwanBusinessIndicator": ["date","leading","leading_notrend","coincident","coincident_notrend","lagging","lagging_notrend","monitoring","monitoring_color"],
            "TaiwanStockTradingDate": ["date"],
        }
        cols = EMPTY_COLUMNS.get(key)
        return pd.DataFrame(columns=cols) if cols else pd.DataFrame()

    # --------- 原本 DataLoader 封裝的便捷方法（以 SQL 取代） ---------
    def taiwan_stock_info(self, timeout: int = None) -> pd.DataFrame:
        return self.get_data(Dataset.TaiwanStockInfo, timeout=timeout)

    def taiwan_stock_info_with_warrant(self, timeout: int = None) -> pd.DataFrame:
        """get 台股總覽(包含權證)
        :param timeout (int): timeout seconds, default None

        :return: 台股總覽 TaiwanStockInfoWithWarrant
        :rtype pd.DataFrame
        :rtype column industry_category (str)
        :rtype column stock_id (str)
        :rtype column stock_name (str)
        :rtype column type (str)
        """
        return self.get_data(
            dataset=Dataset.TaiwanStockInfoWithWarrant,
            timeout=timeout,
        )

    def taiwan_securities_trader_info(
        self, timeout: int = None
    ) -> pd.DataFrame:
        """get 證券商資訊表
        :param timeout (int): timeout seconds, default None

        :return: 證券商資訊表 TaiwanSecuritiesTraderInfo
        :rtype pd.DataFrame
        :rtype column securities_trader_id (str)
        :rtype column securities_trader (str)
        :rtype column date (str)
        :rtype column address (str)
        :rtype column phone (str)
        """
        securities_trader_info = self.get_data(
            dataset=Dataset.TaiwanSecuritiesTraderInfo, timeout=timeout
        )
        return securities_trader_info

    def taiwan_stock_margin_purchase_short_sale(
        self,
        stock_id: str = "",
        start_date: str = "",
        end_date: str = "",
        timeout: int = None,
    ) -> pd.DataFrame:
        """
        get 個股融資融劵表 (SQL 版本)
        對齊 FinMind 介面：Dataset.TaiwanStockMarginPurchaseShortSale
        回傳欄位：
        date, stock_id,
        MarginPurchaseBuy, MarginPurchaseCashRepayment, MarginPurchaseLimit,
        MarginPurchaseSell, MarginPurchaseTodayBalance, MarginPurchaseYesterdayBalance,
        Note, OffsetLoanAndShort,
        ShortSaleBuy, ShortSaleCashRepayment, ShortSaleLimit, ShortSaleSell,
        ShortSaleTodayBalance, ShortSaleYesterdayBalance
        """
        return self.get_data(
            dataset=Dataset.TaiwanStockMarginPurchaseShortSale,
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )



    def taiwan_stock_daily(
        self, stock_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockPrice, data_id=stock_id, start_date=start_date, end_date=end_date, timeout=timeout
        )

    def taiwan_stock_daily_adj(
        self, stock_id: str, start_date: str, end_date: str, timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockPriceAdj, data_id=stock_id, start_date=start_date, end_date=end_date, timeout=timeout
        )

    def taiwan_stock_institutional_investors(
        self, stock_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockInstitutionalInvestorsBuySell,
            data_id=stock_id, start_date=start_date, end_date=end_date, timeout=timeout
        )

    def taiwan_stock_month_revenue(
        self, stock_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockMonthRevenue, data_id=stock_id, start_date=start_date, end_date=end_date, timeout=timeout
        )

    def taiwan_stock_financial_statement(
        self, stock_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        # 注意：這裡回傳 FinMind「long 版」的科目(type/value/origin_name)
        return self.get_data(
            dataset=Dataset.TaiwanStockFinancialStatements, data_id=stock_id, start_date=start_date, end_date=end_date, timeout=timeout
        )

    def taiwan_stock_dividend(
        self, stock_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockDividend, data_id=stock_id, start_date=start_date, end_date=end_date, timeout=timeout
        )


    def taiwan_stock_disposition_securities_period(
        self, stock_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockDispositionSecuritiesPeriod,
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )


    def taiwan_stock_industry_chain(self, stock_id: str = "", timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockIndustryChain,
            stock_id=stock_id,
            timeout=timeout,
        )

    def cnn_fear_greed_index(self, start_date: str = "", end_date: str = "", timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.CnnFearGreedIndex,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_every5seconds_index(self, data_id: str = "", date: str = "", timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockEvery5SecondsIndex,
            data_id=data_id,
            start_date=date,
            timeout=timeout,
        )

    def taiwan_stock_trading_date(self, start_date: str = "", end_date: str = "", timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockTradingDate,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_info_with_warrant_summary(
        self, start_date: str = "", end_date: str = "", data_id: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockInfoWithWarrantSummary,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_split_price(self, start_date: str = "", end_date: str = "", timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockSplitPrice,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_par_value_change(self, start_date: str = "", end_date: str = "", timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockParValueChange,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_convertible_bond_info(self, timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockConvertibleBondInfo,
            timeout=timeout,
        )

    def taiwan_stock_convertible_bond_daily(
        self, cb_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockConvertibleBondDaily,
            data_id=cb_id,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_convertible_bond_institutional_investors(
        self, cb_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockConvertibleBondInstitutionalInvestors,
            data_id=cb_id,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_convertible_bond_daily_overview(
        self, cb_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockConvertibleBondDailyOverview,
            data_id=cb_id,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_stock_margin_short_sale_suspension(
        self, stock_id: str = "", start_date: str = "", end_date: str = "", timeout: int = None
    ) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanStockMarginShortSaleSuspension,
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )

    def taiwan_business_indicator(self, start_date: str = "", end_date: str = "", timeout: int = None) -> pd.DataFrame:
        return self.get_data(
            dataset=Dataset.TaiwanBusinessIndicator,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )




    # --------- 資源釋放 ---------
    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()



class Feature:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_stock_params(self, stock_data: pd.DataFrame):
        stock_data["date"] = stock_data["date"].astype(str)
        stock_id = stock_data["stock_id"].values[0]
        start_date = stock_data["date"].min()
        end_date = stock_data["date"].max()
        return stock_id, start_date, end_date

    def add_kline_institutional_investors(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        stock_id, start_date, end_date = self.get_stock_params(stock_data)
        institutional_investors_df = self.data_loader.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )

        if institutional_investors_df.empty:
            return stock_data  # 沒資料就直接回傳

        # 外資
        fi = institutional_investors_df.query("name == 'Foreign_Investor'")[["date", "buy", "sell"]].copy()
        fi["Foreign_Investor_diff"] = fi["buy"] - fi["sell"]
        fi = fi.drop(["buy", "sell"], axis=1)

        # 投信
        it = institutional_investors_df.query("name == 'Investment_Trust'")[["date", "buy", "sell"]].copy()
        it["Investment_Trust_diff"] = it["buy"] - it["sell"]
        it = it.drop(["buy", "sell"], axis=1)

        stock_data = stock_data.merge(fi, on="date", how="left").merge(it, on="date", how="left")
        return stock_data

    def add_kline_margin_purchase_short_sale(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        stock_id, start_date, end_date = self.get_stock_params(stock_data)
        mpss_df = self.data_loader.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )

        if mpss_df.empty:
            return stock_data  # 沒資料就直接回傳

        # 融資
        mp = mpss_df[["date", "MarginPurchaseBuy", "MarginPurchaseSell"]].copy()
        mp["Margin_Purchase_diff"] = mp["MarginPurchaseBuy"] - mp["MarginPurchaseSell"]
        mp = mp.drop(["MarginPurchaseBuy", "MarginPurchaseSell"], axis=1)

        # 融券
        ss = mpss_df[["date", "ShortSaleBuy", "ShortSaleSell"]].copy()
        ss["Short_Sale_diff"] = ss["ShortSaleBuy"] - ss["ShortSaleSell"]
        ss = ss.drop(["ShortSaleBuy", "ShortSaleSell"], axis=1)

        stock_data = stock_data.merge(mp, on="date", how="left").merge(ss, on="date", how="left")
        return stock_data
